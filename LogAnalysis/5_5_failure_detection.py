"""
5.5 Failure detection — wersja uproszczona

- DETECTION: własna reguła failure_candidate TYLKO z metryk
- EVALUATION: porównanie do EventCode 500/501 i FailureWindow (TP/FP/FN/TN + Prec/Rec/F1)
- WINDOW EVAL: FailureWindow na oknach 1 min (MAX w oknie) per System
- PROPAGATION: TOP par A->B z opóźnieniem 1 okna (A w t -> B w t+1)
"""

from pathlib import Path
import json
import os
import duckdb

THIS_DIR = Path(__file__).resolve().parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"
STATS_JSON = THIS_DIR.parent / "logs.csv.stats.json"

THREADS = 8
MEMORY_LIMIT_GB = 16

SAMPLE_PCT = 5.0
SAMPLE_SIZE = None

K_LAT_SIGMA = 3.0
RETRIES_THRESHOLD = 0.0
NETERR_Q99 = 0.99
DISKQ_Q99 = 0.99

WINDOW_MINUTES = 1
PROP_LAG_MINUTES = 1
PAIR_MIN_SUPPORT = 200
TOP_PAIRS = 10


def load_stats_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def prf1(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def confusion_event_level(con: duckdb.DuckDBPyConnection, pred_expr: str, label_expr: str):
    tp, fp, fn, tn = con.execute(
        f"""
        SELECT
          SUM(CASE WHEN ({pred_expr}) AND ({label_expr}) THEN 1 ELSE 0 END) AS tp,
          SUM(CASE WHEN ({pred_expr}) AND NOT ({label_expr}) THEN 1 ELSE 0 END) AS fp,
          SUM(CASE WHEN NOT ({pred_expr}) AND ({label_expr}) THEN 1 ELSE 0 END) AS fn,
          SUM(CASE WHEN NOT ({pred_expr}) AND NOT ({label_expr}) THEN 1 ELSE 0 END) AS tn
        FROM events
        """
    ).fetchone()
    tp, fp, fn, tn = map(int, (tp or 0, fp or 0, fn or 0, tn or 0))
    p, r, f1 = prf1(tp, fp, fn)
    return tp, fp, fn, tn, p, r, f1


def confusion_window_level(con: duckdb.DuckDBPyConnection, pred_expr: str, label_expr: str, window_minutes: int):
    tp, fp, fn, tn = con.execute(
        f"""
        WITH e AS (
          SELECT
            date_trunc('minute', Timestamp)
              - (EXTRACT(minute FROM Timestamp)::INTEGER % {window_minutes}) * INTERVAL '1 minute' AS win_start,
            SourceSystem,
            CASE WHEN {pred_expr} THEN 1 ELSE 0 END AS pred,
            CASE WHEN {label_expr} THEN 1 ELSE 0 END AS label
          FROM events
        ),
        w AS (
          SELECT
            win_start,
            SourceSystem,
            MAX(pred) AS pred_win,
            MAX(label) AS label_win
          FROM e
          GROUP BY 1,2
        )
        SELECT
          SUM(CASE WHEN pred_win=1 AND label_win=1 THEN 1 ELSE 0 END) AS tp,
          SUM(CASE WHEN pred_win=1 AND label_win=0 THEN 1 ELSE 0 END) AS fp,
          SUM(CASE WHEN pred_win=0 AND label_win=1 THEN 1 ELSE 0 END) AS fn,
          SUM(CASE WHEN pred_win=0 AND label_win=0 THEN 1 ELSE 0 END) AS tn
        FROM w
        """
    ).fetchone()
    tp, fp, fn, tn = map(int, (tp or 0, fp or 0, fn or 0, tn or 0))
    p, r, f1 = prf1(tp, fp, fn)
    return tp, fp, fn, tn, p, r, f1


def main():
    if not PARQUET_DIR.exists():
        raise RuntimeError(f"Brak danych: {PARQUET_DIR}")

    print("=" * 70)
    print("  5.5 FAILURE DETECTION ")
    print("=" * 70)

    con = duckdb.connect()
    con.execute("PRAGMA threads=?", [THREADS])
    con.execute("PRAGMA memory_limit=?", [f"{MEMORY_LIMIT_GB}GB"])

    print("\n[1/4] Wczytywanie danych...")
    total_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)"
    ).fetchone()[0]

    sample_rate = None
    if SAMPLE_PCT and 0 < SAMPLE_PCT <= 100:
        sample_rate = SAMPLE_PCT / 100.0
        print(f"  Próbkowanie: {SAMPLE_PCT:.2f}% z {int(total_rows):,} wierszy")
    elif SAMPLE_SIZE and total_rows > SAMPLE_SIZE:
        sample_rate = SAMPLE_SIZE / total_rows
        print(f"  Próbkowanie: {SAMPLE_SIZE:,} z {int(total_rows):,} ({sample_rate*100:.2f}%)")
    else:
        print(f"  Wszystkie dane: {int(total_rows):,} wierszy")

    sample_predicate = ""
    if sample_rate and 0 < sample_rate < 1:
        sample_predicate = f" AND random() < {sample_rate:.8f}"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE events AS
        SELECT
          Timestamp,
          SourceSystem,
          EventCode,
          COALESCE(FailureWindow,false) AS FailureWindow,
          try_cast(LatencyMs AS DOUBLE) AS LatencyMs,
          try_cast(Retries AS DOUBLE) AS Retries,
          try_cast(DiskQueueLength AS DOUBLE) AS DiskQueueLength,
          try_cast(NetworkErrors AS DOUBLE) AS NetworkErrors
        FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
        WHERE Timestamp IS NOT NULL AND SourceSystem IS NOT NULL AND LatencyMs IS NOT NULL
        {sample_predicate}
        """
    )

    print("\n[2/4] Progi + detekcja failure_candidate...")

    lat_mean, lat_std = con.execute("SELECT AVG(LatencyMs), stddev_samp(LatencyMs) FROM events").fetchone()
    lat_mean, lat_std = float(lat_mean), float(lat_std)
    lat_thr = lat_mean + K_LAT_SIGMA * lat_std

    diskq_thr = con.execute(
        f"SELECT approx_quantile(DiskQueueLength, {DISKQ_Q99}) FROM events WHERE DiskQueueLength IS NOT NULL"
    ).fetchone()[0]
    diskq_thr = float(diskq_thr) if diskq_thr is not None else None

    neterr_thr = con.execute(
        f"SELECT approx_quantile(NetworkErrors, {NETERR_Q99}) FROM events WHERE NetworkErrors IS NOT NULL"
    ).fetchone()[0]
    neterr_thr = float(neterr_thr) if neterr_thr is not None else None

    print(f"  Latency thr = mean+{K_LAT_SIGMA:g}σ = {lat_thr:.2f} ms")
    print(f"  DiskQueueLength thr = p{int(DISKQ_Q99*100)} = {diskq_thr:.2f}" if diskq_thr is not None else "  DiskQueueLength thr = (brak)")
    print(f"  NetworkErrors thr = p{int(NETERR_Q99*100)} = {neterr_thr:.2f}" if neterr_thr is not None else "  NetworkErrors thr = (brak)")

    diskq_part = f"(DiskQueueLength IS NOT NULL AND DiskQueueLength > {diskq_thr})" if diskq_thr is not None else "FALSE"
    neterr_part = f"(NetworkErrors IS NOT NULL AND NetworkErrors > {neterr_thr})" if neterr_thr is not None else "FALSE"
    pred_failure = f"""
      (LatencyMs > {lat_thr})
      AND (
        (Retries IS NOT NULL AND Retries > {RETRIES_THRESHOLD})
        OR {neterr_part}
        OR {diskq_part}
      )
    """.strip()

    n_cand, n_total = con.execute(
        f"SELECT SUM(CASE WHEN {pred_failure} THEN 1 ELSE 0 END), COUNT(*) FROM events"
    ).fetchone()
    n_cand, n_total = int(n_cand or 0), int(n_total or 0)
    print(f"  failure_candidate: {n_cand:,} / {n_total:,}  ({(n_cand/n_total*100 if n_total else 0):.3f}%)")

    print("\n[3/4] Porównanie z labelami...")

    rows = []
    tp, fp, fn, tn, p, r, f1 = confusion_event_level(con, pred_failure, "EventCode IN (500,501)")
    rows.append(("EventCode 500/501 (event)", p, r, f1, tp, fp, fn, tn))

    tp, fp, fn, tn, p, r, f1 = confusion_event_level(con, pred_failure, "FailureWindow")
    rows.append(("FailureWindow (event)", p, r, f1, tp, fp, fn, tn))

    tp, fp, fn, tn, p, r, f1 = confusion_window_level(con, pred_failure, "FailureWindow", WINDOW_MINUTES)
    rows.append((f"FailureWindow (window {WINDOW_MINUTES}m, per System)", p, r, f1, tp, fp, fn, tn))

    print(f"{'Label':<40} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TP':>10} {'FP':>10} {'FN':>10} {'TN':>10}")
    print("-" * 105)
    for label, p, r, f1, tp, fp, fn, tn in rows:
        print(f"{label:<40} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {tp:>10,} {fp:>10,} {fn:>10,} {tn:>10,}")

    stats = load_stats_json(STATS_JSON)
    print("\n[stats.json]")
    if stats is None:
        print("  Brak / nie można odczytać -> pomijam.")
    else:
        fe = stats.get("FailureEvents")
        efw = stats.get("ExpectedFailureWindows")
        ec_cnt, fw_cnt = con.execute(
            "SELECT SUM(CASE WHEN EventCode IN (500,501) THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN FailureWindow THEN 1 ELSE 0 END) FROM events"
        ).fetchone()
        ec_cnt = int(ec_cnt or 0)
        fw_cnt = int(fw_cnt or 0)
        print(f"  FailureEvents (stats.json): {fe if fe is not None else 'N/A'}")
        if efw is not None:
            print(f"  ExpectedFailureWindows: {efw}")
        print(f"  EventCode 500/501 count: {ec_cnt:,}")
        print(f"  FailureWindow count:      {fw_cnt:,}")
        print(f"  FailureCandidate count:   {n_cand:,}")
        if isinstance(fe, int):
            diff = n_cand - fe
            print(f"  Diff (candidate - stats): {diff:+,}")

    print("\n[4/4] Propagacja (System, lag=1m)...")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW sys_min AS
        SELECT
          SourceSystem AS System,
          date_trunc('minute', Timestamp) AS minute,
          MAX(CASE WHEN {pred_failure} THEN 1 ELSE 0 END) AS fail_pred
        FROM events
        GROUP BY 1, 2
        """
    )

    pairs = con.execute(
        f"""
        WITH a AS (
          SELECT System AS A, minute AS t
          FROM sys_min
          WHERE fail_pred = 1
        ),
        b AS (
          SELECT System AS B, minute AS t, fail_pred AS b_fail
          FROM sys_min
        ),
        base AS (
          SELECT System AS B, AVG(fail_pred) AS base_rate
          FROM sys_min
          GROUP BY 1
        )
        SELECT
          A,
          B,
          COUNT(*) AS n,
          SUM(CASE WHEN b.b_fail=1 THEN 1 ELSE 0 END) AS b_fail,
          (SUM(CASE WHEN b.b_fail=1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) AS p_fail,
          base.base_rate AS base_rate
        FROM a
        JOIN b ON b.t = a.t + INTERVAL '{PROP_LAG_MINUTES} minute'
        JOIN base USING(B)
        WHERE A <> B
        GROUP BY 1, 2, 6
        HAVING COUNT(*) >= {PAIR_MIN_SUPPORT}
        ORDER BY p_fail DESC, n DESC
        LIMIT {TOP_PAIRS}
        """
    ).fetchall()

    print(f"  TOP {TOP_PAIRS} par (support>={PAIR_MIN_SUPPORT})")
    print(f"  {'A':<20} {'B':<20} {'n':>10} {'b_fail':>10} {'P(fail|A)':>12} {'P(fail B)':>10}")
    print("  " + "-" * 89)
    for A, B, n, b_fail, p_fail, base_rate in pairs:
        print(f"  {str(A):<20} {str(B):<20} {int(n):>10,} {int(b_fail):>10,} {float(p_fail):>12.3f} {float(base_rate):>10.3f}")


if __name__ == "__main__":
    main()
