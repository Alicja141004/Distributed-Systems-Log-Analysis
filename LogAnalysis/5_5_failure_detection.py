from pathlib import Path
import json
import duckdb

THIS_DIR = Path(__file__).resolve().parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"
STATS_JSON = THIS_DIR.parent / "logs.csv.stats.json"

THREADS = 8
MEMORY_LIMIT_GB = 16

# Stałe/konfiguracja analizy
RETRIES_THRESHOLD = 0
PROP_LAG_MINUTES = 1

# grid search
TUNE_LABEL = "FailureWindow"
TUNE_LAT_PCTS = [0.95, 0.97, 0.99]
TUNE_NETDISK_PCTS = [0.90, 0.95, 0.97]
TUNE_MIN_SIGNALS = [1, 2]

# Sampling dla bardzo dużych danych
SAMPLE_MAX_ROWS = 5_000_000
SAMPLE_PCT_LARGE = 5.0


def get_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    cols = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {c[1] for c in cols}


def load_stats_json(path: Path):
    # Stats JSON z generatora (referencja do porównań liczebności)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def prf1(tp: int, fp: int, fn: int):
    # Klasyczne metryki jakości dla porównań z labelami
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def confusion_event_level(
    con: duckdb.DuckDBPyConnection, pred_expr: str, label_expr: str, table: str = "events_enriched"
):
    # Macierz pomyłek na poziomie eventów
    tp, fp, fn, tn = con.execute(
        f"""
        SELECT
          SUM(CASE WHEN ({pred_expr}) AND ({label_expr}) THEN 1 ELSE 0 END) AS tp,
          SUM(CASE WHEN ({pred_expr}) AND NOT ({label_expr}) THEN 1 ELSE 0 END) AS fp,
          SUM(CASE WHEN NOT ({pred_expr}) AND ({label_expr}) THEN 1 ELSE 0 END) AS fn,
          SUM(CASE WHEN NOT ({pred_expr}) AND NOT ({label_expr}) THEN 1 ELSE 0 END) AS tn
        FROM {table}
        """
    ).fetchone()
    tp, fp, fn, tn = map(int, (tp or 0, fp or 0, fn or 0, tn or 0))
    p, r, f1 = prf1(tp, fp, fn)
    return tp, fp, fn, tn, p, r, f1


def main():
    if not PARQUET_DIR.exists():
        raise RuntimeError(f"Brak danych: {PARQUET_DIR}")

    stats = load_stats_json(STATS_JSON)

    print("=" * 120)
    print(" 5.5 FAILURE DETECTION")
    print("=" * 120)

    con = duckdb.connect()
    con.execute("PRAGMA threads=?", [THREADS])
    con.execute("PRAGMA memory_limit=?", [f"{MEMORY_LIMIT_GB}GB"])

    # Load danych z parquetów
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE events AS
        SELECT
          Timestamp,
          SourceSystem,
          CorrelationId,
          EventCode,
          COALESCE(FailureWindow,false) AS FailureWindow,
          try_cast(LatencyMs AS DOUBLE) AS LatencyMs,
          try_cast(Retries AS DOUBLE) AS Retries,
          try_cast(DiskQueueLength AS DOUBLE) AS DiskQueueLength,
          try_cast(NetworkErrors AS DOUBLE) AS NetworkErrors
        FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
        WHERE Timestamp IS NOT NULL AND SourceSystem IS NOT NULL
        """
    )

    total_events = int(con.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    sample_rate = None
    if total_events > SAMPLE_MAX_ROWS:
        sample_rate = SAMPLE_PCT_LARGE / 100.0

    # Split czasowy: train / validation (np. 80/20)
    split_ratio = 0.8
    cutoff_ts = con.execute(
        """
        WITH c AS (SELECT COUNT(*) AS n FROM events),
        o AS (
          SELECT Timestamp
          FROM events
          ORDER BY Timestamp
          LIMIT 1
          OFFSET (SELECT CAST(n * ? AS BIGINT) FROM c)
        )
        SELECT Timestamp FROM o
        """,
        [split_ratio],
    ).fetchone()[0]
    cutoff_ts_sql = f"TIMESTAMP '{cutoff_ts}'"

    # Baseline per system + progi per system (na train)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW events_train AS
        SELECT * FROM events WHERE Timestamp < {cutoff_ts_sql}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW events_valid AS
        SELECT * FROM events WHERE Timestamp >= {cutoff_ts_sql}
        """
    )

    train_sample_pred = ""
    valid_sample_pred = ""
    if sample_rate and 0 < sample_rate < 1:
        train_sample_pred = f" AND random() < {sample_rate:.8f}"
        valid_sample_pred = f" AND random() < {sample_rate:.8f}"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE events_train_s AS
        SELECT * FROM events_train WHERE 1=1 {train_sample_pred}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE events_valid_s AS
        SELECT * FROM events_valid WHERE 1=1 {valid_sample_pred}
        """
    )

    ec_cnt, fw_cnt = con.execute(
        """
        SELECT
          SUM(CASE WHEN EventCode IN (500,501) THEN 1 ELSE 0 END) AS ec,
          SUM(CASE WHEN FailureWindow THEN 1 ELSE 0 END) AS fw
        FROM events_valid_s
        """
    ).fetchone()
    ec_cnt, fw_cnt = int(ec_cnt or 0), int(fw_cnt or 0)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW sys_thresholds AS
        SELECT
          SourceSystem,
          AVG(LatencyMs) AS lat_mean,
          stddev_samp(LatencyMs) AS lat_std,
          approx_quantile(LatencyMs, 0.5) AS lat_med,
          approx_quantile(LatencyMs, 0.90) AS lat_p90,
          approx_quantile(LatencyMs, 0.97) AS lat_p97,
          approx_quantile(LatencyMs, 0.95) AS lat_p95,
          approx_quantile(LatencyMs, 0.99) AS lat_p99,

          AVG(NetworkErrors) AS neterr_mean,
          stddev_samp(NetworkErrors) AS neterr_std,
          approx_quantile(NetworkErrors, 0.5) AS neterr_med,
          approx_quantile(NetworkErrors, 0.90) AS neterr_p90,
          approx_quantile(NetworkErrors, 0.97) AS neterr_p97,
          approx_quantile(NetworkErrors, 0.95) AS neterr_p95,
          approx_quantile(NetworkErrors, 0.99) AS neterr_p99,

          AVG(DiskQueueLength) AS diskq_mean,
          stddev_samp(DiskQueueLength) AS diskq_std,
          approx_quantile(DiskQueueLength, 0.5) AS diskq_med,
          approx_quantile(DiskQueueLength, 0.90) AS diskq_p90,
          approx_quantile(DiskQueueLength, 0.97) AS diskq_p97,
          approx_quantile(DiskQueueLength, 0.95) AS diskq_p95,
          approx_quantile(DiskQueueLength, 0.99) AS diskq_p99
        FROM events_train_s
        GROUP BY 1
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW events_enriched AS
        SELECT
          e.*,
          t.lat_mean, t.lat_std, t.lat_med, t.lat_p90, t.lat_p95, t.lat_p97, t.lat_p99,
          t.neterr_mean, t.neterr_std, t.neterr_med, t.neterr_p90, t.neterr_p95, t.neterr_p97, t.neterr_p99,
          t.diskq_mean, t.diskq_std, t.diskq_med, t.diskq_p90, t.diskq_p95, t.diskq_p97, t.diskq_p99
        FROM events e
        JOIN sys_thresholds t USING(SourceSystem)
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW events_enriched_train_s AS
        SELECT
          e.*,
          t.lat_mean, t.lat_std, t.lat_med, t.lat_p90, t.lat_p95, t.lat_p97, t.lat_p99,
          t.neterr_mean, t.neterr_std, t.neterr_med, t.neterr_p90, t.neterr_p95, t.neterr_p97, t.neterr_p99,
          t.diskq_mean, t.diskq_std, t.diskq_med, t.diskq_p90, t.diskq_p95, t.diskq_p97, t.diskq_p99
        FROM events_train_s e
        JOIN sys_thresholds t USING(SourceSystem)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW events_enriched_valid_s AS
        SELECT
          e.*,
          t.lat_mean, t.lat_std, t.lat_med, t.lat_p90, t.lat_p95, t.lat_p97, t.lat_p99,
          t.neterr_mean, t.neterr_std, t.neterr_med, t.neterr_p90, t.neterr_p95, t.neterr_p97, t.neterr_p99,
          t.diskq_mean, t.diskq_std, t.diskq_med, t.diskq_p90, t.diskq_p95, t.diskq_p97, t.diskq_p99
        FROM events_valid_s e
        JOIN sys_thresholds t USING(SourceSystem)
        """
    )

    # Reguła failure_candidate (event-level) + auto strojenie
    def label_expr_for(name: str) -> str:
        if name == "EventCode 500/501":
            return "EventCode IN (500,501)"
        return "FailureWindow"

    tune_label_expr = label_expr_for(TUNE_LABEL)

    def pct_col(prefix: str, pct: float) -> str:
        if pct >= 0.99:
            return f"{prefix}_p99"
        if pct >= 0.97:
            return f"{prefix}_p97"
        if pct >= 0.95:
            return f"{prefix}_p95"
        if pct >= 0.90:
            return f"{prefix}_p90"
        return f"{prefix}_p90"

    def build_expr(lat_pct: float, nd_pct: float, min_signals: int) -> str:
        lat_c = pct_col("lat", lat_pct)
        net_c = pct_col("neterr", nd_pct)
        disk_c = pct_col("diskq", nd_pct)
        return (
            f"LatencyMs >= {lat_c} AND ("
            f"(CASE WHEN Retries > {RETRIES_THRESHOLD} THEN 1 ELSE 0 END"
            f" + CASE WHEN NetworkErrors >= {net_c} THEN 1 ELSE 0 END"
            f" + CASE WHEN DiskQueueLength >= {disk_c} THEN 1 ELSE 0 END) >= {min_signals}"
            f")"
        )

    tuned = []
    for lat_pct in TUNE_LAT_PCTS:
        for nd_pct in TUNE_NETDISK_PCTS:
            for min_signals in TUNE_MIN_SIGNALS:
                expr = build_expr(lat_pct, nd_pct, min_signals)
                tp, fp, fn, tn, p, r, f1 = confusion_event_level(
                    con, expr, tune_label_expr, table="events_enriched_train_s"
                )
                cnt = int(
                    con.execute(
                        f"SELECT SUM(CASE WHEN {expr} THEN 1 ELSE 0 END) FROM events_enriched_train_s"
                    ).fetchone()[0]
                    or 0
                )
                tuned.append(
                    {
                        "lat_pct": lat_pct,
                        "nd_pct": nd_pct,
                        "min_signals": min_signals,
                        "expr": expr,
                        "cnt": cnt,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "p": p,
                        "r": r,
                        "f1": f1,
                    }
                )

    tuned.sort(key=lambda x: (x["f1"], x["r"], -x["cnt"]), reverse=True)
    best = tuned[0]

    rules = [
        (
            f"Auto-tuned vs {TUNE_LABEL}: "
            f"lat>={best['lat_pct']:.2f}p, net/disk>={best['nd_pct']:.2f}p, "
            f"min_signals>={best['min_signals']}",
            best["expr"],
        ),
    ]

    print("\n[1] failure_candidate — wybrana reguła:")
    if sample_rate and 0 < sample_rate < 1:
        print(f"  Sampling: {SAMPLE_PCT_LARGE:.1f}% (train/valid)")
    print("  Reguła:")
    print(f"    LatencyMs >= p{int(best['lat_pct']*100)} per system")
    print(f"    AND co najmniej {best['min_signals']} z:")
    print(f"      - Retries > {RETRIES_THRESHOLD}")
    print(f"      - NetworkErrors >= p{int(best['nd_pct']*100)} per system")
    print(f"      - DiskQueueLength >= p{int(best['nd_pct']*100)} per system")
    rule_counts = []
    for desc, expr in rules:
        cnt = int(
            con.execute(f"SELECT SUM(CASE WHEN {expr} THEN 1 ELSE 0 END) FROM events_enriched_valid_s").fetchone()[0]
            or 0
        )
        rule_counts.append((desc, expr, cnt))

    pred_failure = rule_counts[0][1]
    cand_cnt = rule_counts[0][2]

    # Porównanie liczebności (count-level)
    print("\n[2] Porównanie liczebności (count-level) [validation]:")
    if sample_rate and 0 < sample_rate < 1:
        print(f"  (uwaga: sampling {SAMPLE_PCT_LARGE:.1f}%, liczby ~% pełnych danych)")
    print("  failure_candidate:")
    print(f"    {cand_cnt:,}")
    print("  Referencje:")
    refs = [
        ("EventCode 500/501", ec_cnt),
        ("FailureWindow", fw_cnt),
    ]
    st_fail = int(stats.get("FailureEvents", 0) or 0) if isinstance(stats, dict) else None
    if st_fail is not None:
        refs.append(("FailureEvents (stats)", st_fail))

    print(f"{'Referencja':<24} {'Count':>12} {'Diff':>12} {'AbsDiff':>12}")
    print("-" * 62)
    closest = None
    for name, count in refs:
        diff = cand_cnt - count
        absdiff = abs(diff)
        print(f"{name:<24} {count:>12,} {diff:>12,} {absdiff:>12,}")
        if closest is None or absdiff < closest[1]:
            closest = (name, absdiff)
    if closest:
        print(f"  -> Najbliżej do: {closest[0]}")

    # Porównanie globalne (event-level)
    print("\n[3] Porównanie globalne (event-level) [validation]:")
    print(f"{'Label':<25} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TP':>13} {'FP':>13} {'FN':>13} {'TN':>13}")
    print("-" * 111)
    for label_name, label_expr in [
        ("EventCode 500/501", "EventCode IN (500,501)"),
        ("FailureWindow", "FailureWindow"),
    ]:
        tp, fp, fn, tn, p, r, f1 = confusion_event_level(
            con, pred_failure, label_expr, table="events_enriched_valid_s"
        )
        print(f"{label_name:<25} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {tp:>13,} {fp:>13,} {fn:>13,} {tn:>13,}")

    # Breakdown per system (vs FailureWindow)
    print("\n[4] Breakdown per system (metrycznie: jak działa na każdym systemie vs FailureWindow) [validation]:")
    print(f"{'System':<22} {'cand':>10} {'fw':>10} {'TP':>10} {'FP':>10} {'FN':>10} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 102)
    per_sys = con.execute(
        f"""
        WITH s AS (
          SELECT
            SourceSystem AS System,
            SUM(CASE WHEN {pred_failure} THEN 1 ELSE 0 END) AS cand,
            SUM(CASE WHEN FailureWindow THEN 1 ELSE 0 END) AS fw,

            SUM(CASE WHEN ({pred_failure}) AND FailureWindow THEN 1 ELSE 0 END) AS tp,
            SUM(CASE WHEN ({pred_failure}) AND NOT FailureWindow THEN 1 ELSE 0 END) AS fp,
            SUM(CASE WHEN NOT ({pred_failure}) AND FailureWindow THEN 1 ELSE 0 END) AS fn
          FROM events_enriched_valid_s
          GROUP BY 1
        )
        SELECT
          System,
          cand, fw,
          tp, fp, fn,
          (tp * 1.0 / NULLIF(tp + fp, 0)) AS prec,
          (tp * 1.0 / NULLIF(tp + fn, 0)) AS rec,
          (2 * (tp * 1.0 / NULLIF(tp + fp, 0)) * (tp * 1.0 / NULLIF(tp + fn, 0))
             / NULLIF((tp * 1.0 / NULLIF(tp + fp, 0)) + (tp * 1.0 / NULLIF(tp + fn, 0)), 0)
          ) AS f1
        FROM s
        ORDER BY f1 DESC NULLS LAST, fw DESC
        """
    ).fetchall()

    for System, cand, fw, tp, fp, fn, prec, rec, f1 in per_sys:
        prec = float(prec) if prec is not None else 0.0
        rec = float(rec) if rec is not None else 0.0
        f1 = float(f1) if f1 is not None else 0.0
        print(f"{str(System):<22} {int(cand):>10,} {int(fw):>10,} {int(tp):>10,} {int(fp):>10,} {int(fn):>10,} "
              f"{prec:>7.3f} {rec:>7.3f} {f1:>7.3f}")

    # Propagacja failure między systemami (A -> B w obrębie CorrelationId)
    print("\n[5] Propagacja failure między systemami:")
    cols = get_columns(con, "events")
    if "CorrelationId" in cols:
        # Sekwencje A -> B w oknie czasowym (lag)
        pairs = con.execute(
            f"""
            WITH f AS (
              SELECT
                CorrelationId,
                SourceSystem,
                Timestamp
              FROM events_enriched_valid_s
              WHERE CorrelationId IS NOT NULL AND {pred_failure}
            ),
            ordered AS (
              SELECT
                CorrelationId,
                SourceSystem AS curr_sys,
                Timestamp AS curr_ts,
                LEAD(SourceSystem) OVER (PARTITION BY CorrelationId ORDER BY Timestamp, SourceSystem) AS next_sys,
                LEAD(Timestamp) OVER (PARTITION BY CorrelationId ORDER BY Timestamp, SourceSystem) AS next_ts
              FROM f
            ),
            edges AS (
              SELECT
                curr_sys AS a_sys,
                next_sys AS b_sys
              FROM ordered
              WHERE next_sys IS NOT NULL
                AND curr_sys <> next_sys
                AND next_ts <= curr_ts + INTERVAL '{PROP_LAG_MINUTES} minute'
            )
            SELECT a_sys, b_sys, COUNT(*) AS n
            FROM edges
            GROUP BY 1,2
            ORDER BY n DESC
            """
        ).fetchall()
        print(f"  tryb: sekwencja w CorrelationId + lag={PROP_LAG_MINUTES}m (top 10)")
    else:
        pairs = con.execute(
            f"""
            WITH e AS (
              SELECT
                date_trunc('minute', Timestamp) AS win_start,
                SourceSystem,
                MAX(CASE WHEN {pred_failure} THEN 1 ELSE 0 END) AS pred_win
              FROM events_enriched_valid_s
              GROUP BY 1,2
            ),
            pairs AS (
              SELECT
                a.SourceSystem AS a_sys,
                b.SourceSystem AS b_sys,
                COUNT(*) AS n
              FROM e a
              JOIN e b
                ON a.win_start = b.win_start
               AND a.SourceSystem <> b.SourceSystem
               AND a.pred_win = 1
               AND b.pred_win = 1
              GROUP BY 1,2
            )
            SELECT * FROM pairs ORDER BY n DESC
            """
        ).fetchall()
        print(f"  tryb: okna 1m (bez CorrelationId, heurystyka)")

    if pairs:
        print(f"  {'A':<20} {'B':<20} {'n':>8}")
        print(f"  {'-'*20} {'-'*20} {'-'*8}")
        for a_sys, b_sys, n in pairs[:10]:
            print(f"  {a_sys:<20} {b_sys:<20} {n:>8,}")
    else:
        print("  Brak par spełniających próg.")

    if pairs:
        # Najczęstsze systemy jako "powodujące" (A) i "dotknięte" (B) w top-10
        top_a = {}
        top_b = {}
        for a_sys, b_sys, n in pairs[:10]:
            top_a[a_sys] = top_a.get(a_sys, 0) + int(n)
            top_b[b_sys] = top_b.get(b_sys, 0) + int(n)
        top_a_sorted = sorted(top_a.items(), key=lambda x: x[1], reverse=True)[:2]
        top_b_sorted = sorted(top_b.items(), key=lambda x: x[1], reverse=True)[:2]

        print("\n  Najczęstsze systemy jako 'powodujące' (A):")
        for sys, cnt in top_a_sorted:
            print(f"    {sys}: {cnt:,}")

        print("  Najczęstsze systemy jako 'dotknięte' (B):")
        for sys, cnt in top_b_sorted:
            print(f"    {sys}: {cnt:,}")


if __name__ == "__main__":
    main()
