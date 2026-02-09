"""
5.3 Wykrywanie spike'ów (bez użycia flag IsSpike* i SpikeWindow)

Wykrywanie spike'ów wyłącznie na podstawie metryk:
1) Metoda progowa:
   - oblicz medianę/średnią LatencyMs i odchylenie standardowe
   - spike: > 3*median lub > mean + k*sigma
   - analogicznie dla CpuUsage / LocalQps

2) Metoda okien czasowych:
   - zbuduj okna czasowe (np. 1 min, 5 min)
   - policz QPS w każdym oknie (tu: AVG(LocalQps) w oknie)
   - okna gdzie QPS > 2.5× poprzedniego okna uznaj za spike

3) Na końcu porównaj wynik z:
   - IsSpikeByLatency
   - IsSpikeByCpu
   - IsSpikeByQps
   - policz precision/recall/F1 (+ tutaj również TN dla pełności)

================================================================================
METODA WŁASNA:
A) Progowa robust: Robust Z-score (median + MAD), spike gdy z > K
B) Okienkowa robust: w oknie 1/5 min spike gdy udział pred >= PCT_THRESHOLD
   + liczymy TP/FP/FN/TN na oknach
================================================================================
"""

from __future__ import annotations

from pathlib import Path
import os
import duckdb

THIS_DIR = Path(__file__).resolve().parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"

K_SIGMA = 3.0
QPS_WINDOW_MULTIPLIER = 2.5
THREADS = 8
MEMORY_LIMIT_GB = 16

SAMPLE_PCT = 5.0
SAMPLE_SIZE = None

ENABLE_BONUS_METHOD = True 

K_LAT_ROBUST = 6.0
K_CPU_ROBUST = 5.0
K_QPS_ROBUST = 6.0

ROBUST_WINDOW_PCT_THRESHOLD = 0.10  
ROBUST_WINDOWS_MINUTES = (1, 5)


def count_rule(con: duckdb.DuckDBPyConnection, name: str, expr: str):
    total, n = con.execute(
        f"""
        SELECT
          COUNT(*) AS total,
          SUM(CASE WHEN {expr} THEN 1 ELSE 0 END) AS n
        FROM events
        """
    ).fetchone()
    total, n = int(total), int(n)
    rate = (n / total) if total else 0.0
    return {"name": name, "count": n, "total": total, "rate": rate}


def count_qps_window(con: duckdb.DuckDBPyConnection, window_minutes: int, multiplier: float):
    spike_windows, spike_events, total_events = con.execute(
        f"""
        WITH events_w AS (
          SELECT
            LocalQps,
            date_trunc('minute', Timestamp)
              - (EXTRACT(minute FROM Timestamp)::INTEGER % {window_minutes}) * INTERVAL '1 minute' AS win_start
          FROM events
        ),
        win_stats AS (
          SELECT
            win_start,
            AVG(LocalQps) AS avg_qps,
            COUNT(*) AS n_events
          FROM events_w
          GROUP BY 1
        ),
        win_lag AS (
          SELECT
            win_start,
            avg_qps,
            n_events,
            LAG(avg_qps) OVER (ORDER BY win_start) AS prev_qps
          FROM win_stats
        )
        SELECT
          SUM(CASE WHEN prev_qps IS NOT NULL AND avg_qps > {multiplier} * prev_qps THEN 1 ELSE 0 END) AS spike_windows,
          SUM(CASE WHEN prev_qps IS NOT NULL AND avg_qps > {multiplier} * prev_qps THEN n_events ELSE 0 END) AS spike_events,
          SUM(n_events) AS total_events
        FROM win_lag
        """
    ).fetchone()

    spike_windows = int(spike_windows or 0)
    spike_events = int(spike_events or 0)
    total_events = int(total_events or 0)
    rate = (spike_events / total_events) if total_events else 0.0
    return {
        "name": f"QPS okna {window_minutes} min (>{multiplier}× poprz.)",
        "spike_windows": spike_windows,
        "spike_events": spike_events,
        "total_events": total_events,
        "rate": rate,
    }


def prf1_event_level(con: duckdb.DuckDBPyConnection, pred_expr: str, label_col: str):
    """Precision/Recall/F1 (+TN) na poziomie eventów: pred_expr vs label_col (flag)."""
    tp, fp, fn, tn = con.execute(
        f"""
        WITH x AS (
          SELECT
            CASE WHEN {pred_expr} THEN 1 ELSE 0 END AS pred,
            CASE WHEN {label_col} THEN 1 ELSE 0 END AS label
          FROM events_eval
        )
        SELECT
          SUM(CASE WHEN pred=1 AND label=1 THEN 1 ELSE 0 END) AS tp,
          SUM(CASE WHEN pred=1 AND label=0 THEN 1 ELSE 0 END) AS fp,
          SUM(CASE WHEN pred=0 AND label=1 THEN 1 ELSE 0 END) AS fn,
          SUM(CASE WHEN pred=0 AND label=0 THEN 1 ELSE 0 END) AS tn
        FROM x
        """
    ).fetchone()

    tp, fp, fn, tn = map(int, (tp or 0, fp or 0, fn or 0, tn or 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return tp, fp, fn, tn, precision, recall, f1


def prf1_qps_windows(con: duckdb.DuckDBPyConnection, window_minutes: int, multiplier: float):
    """
    Precision/Recall/F1 (+TN) na poziomie OKIEN:
    - pred: avg_qps okna > multiplier * avg_qps poprzedniego okna
    - label: MAX(IsSpikeByQps) w oknie
    """
    row = con.execute(
        f"""
        WITH win AS (
          SELECT
            date_trunc('minute', Timestamp)
              - (EXTRACT(minute FROM Timestamp)::INTEGER % {window_minutes}) * INTERVAL '1 minute' AS win_start,
            AVG(LocalQps) AS avg_qps,
            MAX(CASE WHEN IsSpikeByQps THEN 1 ELSE 0 END) AS label
          FROM events_eval
          GROUP BY 1
        ),
        lagged AS (
          SELECT
            win_start,
            avg_qps,
            label,
            LAG(avg_qps) OVER (ORDER BY win_start) AS prev_qps
          FROM win
        ),
        scored AS (
          SELECT
            CASE WHEN prev_qps IS NOT NULL AND avg_qps > {multiplier} * prev_qps THEN 1 ELSE 0 END AS pred,
            label
          FROM lagged
          WHERE prev_qps IS NOT NULL
        )
        SELECT
          COUNT(*) AS total,
          SUM(CASE WHEN pred=1 AND label=1 THEN 1 ELSE 0 END) AS tp,
          SUM(CASE WHEN pred=1 AND label=0 THEN 1 ELSE 0 END) AS fp,
          SUM(CASE WHEN pred=0 AND label=1 THEN 1 ELSE 0 END) AS fn
        FROM scored
        """
    ).fetchone()

    total, tp, fp, fn = map(int, (row[0] or 0, row[1] or 0, row[2] or 0, row[3] or 0))
    tn = total - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return total, tp, fp, fn, tn, precision, recall, f1

def robust_stats_mad(con: duckdb.DuckDBPyConnection):
    """
    Zwraca (mediana, robust_sigma) dla Latency/Cpu/Qps.
    robust_sigma = 1.4826 * MAD, gdzie MAD = median(|x - median(x)|)
    """
    row = con.execute(
        """
        WITH base AS (
          SELECT
            approx_quantile(LatencyMs, 0.5) AS lat_m,
            approx_quantile(CpuUsage, 0.5) AS cpu_m,
            approx_quantile(LocalQps, 0.5) AS qps_m
          FROM events
        ),
        mad AS (
          SELECT
            approx_quantile(ABS(LatencyMs - (SELECT lat_m FROM base)), 0.5) AS lat_mad,
            approx_quantile(ABS(CpuUsage  - (SELECT cpu_m FROM base)), 0.5) AS cpu_mad,
            approx_quantile(ABS(LocalQps  - (SELECT qps_m FROM base)), 0.5) AS qps_mad
          FROM events
        )
        SELECT
          (SELECT lat_m FROM base) AS lat_m,
          1.4826 * (SELECT lat_mad FROM mad) AS lat_sig_r,
          (SELECT cpu_m FROM base) AS cpu_m,
          1.4826 * (SELECT cpu_mad FROM mad) AS cpu_sig_r,
          (SELECT qps_m FROM base) AS qps_m,
          1.4826 * (SELECT qps_mad FROM mad) AS qps_sig_r
        """
    ).fetchone()
    lat_m, lat_sig_r, cpu_m, cpu_sig_r, qps_m, qps_sig_r = map(float, row)
    return (lat_m, lat_sig_r), (cpu_m, cpu_sig_r), (qps_m, qps_sig_r)


def prf1_robust_windows(
    con: duckdb.DuckDBPyConnection,
    window_minutes: int,
    pred_expr: str,
    label_col: str,
    pct_threshold: float,
):
    """
    BONUS: Okienkowa ewaluacja robust (+TN):
    - pred_expr: predykcja event-level (np. robust_z > K)
    - pred okna: AVG(pred) >= pct_threshold
    - label okna: MAX(label_col)=1
    """
    row = con.execute(
        f"""
        WITH e AS (
          SELECT
            date_trunc('minute', Timestamp)
              - (EXTRACT(minute FROM Timestamp)::INTEGER % {window_minutes}) * INTERVAL '1 minute' AS win_start,
            CASE WHEN {pred_expr} THEN 1 ELSE 0 END AS pred,
            CASE WHEN {label_col} THEN 1 ELSE 0 END AS label
          FROM events_eval
        ),
        w AS (
          SELECT
            win_start,
            AVG(pred) AS pred_rate,
            MAX(label) AS label
          FROM e
          GROUP BY 1
        ),
        s AS (
          SELECT
            CASE WHEN pred_rate >= {pct_threshold} THEN 1 ELSE 0 END AS pred,
            label
          FROM w
        )
        SELECT
          COUNT(*) AS total,
          SUM(CASE WHEN pred=1 AND label=1 THEN 1 ELSE 0 END) AS tp,
          SUM(CASE WHEN pred=1 AND label=0 THEN 1 ELSE 0 END) AS fp,
          SUM(CASE WHEN pred=0 AND label=1 THEN 1 ELSE 0 END) AS fn
        FROM s
        """
    ).fetchone()

    total, tp, fp, fn = map(int, (row[0] or 0, row[1] or 0, row[2] or 0, row[3] or 0))
    tn = total - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return total, tp, fp, fn, tn, precision, recall, f1


def main():
    if not PARQUET_DIR.exists():
        raise RuntimeError(f"Brak danych: {PARQUET_DIR}")

    print("=" * 70)
    print("  5.3 WYKRYWANIE SPIKE'ÓW (bez flag IsSpike* i SpikeWindow)")
    print("=" * 70)

    con = duckdb.connect()
    con.execute("PRAGMA threads=?", [THREADS])
    con.execute("PRAGMA memory_limit=?", [f"{MEMORY_LIMIT_GB}GB"])

    print("\n[1/5] Wczytywanie danych...")

    total_rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
        WHERE LatencyMs IS NOT NULL AND CpuUsage IS NOT NULL AND LocalQps IS NOT NULL
        """
    ).fetchone()[0]

    sample_rate = None
    if SAMPLE_PCT and 0 < SAMPLE_PCT <= 100:
        sample_rate = SAMPLE_PCT / 100.0
        print(f"  Próbkowanie: {SAMPLE_PCT:.2f}% z {total_rows:,} wierszy")
    elif SAMPLE_SIZE and total_rows > SAMPLE_SIZE:
        sample_rate = SAMPLE_SIZE / total_rows
        print(f"  Próbkowanie: {SAMPLE_SIZE:,} z {total_rows:,} wierszy ({sample_rate*100:.2f}%)")
    else:
        print(f"  Wszystkie dane: {total_rows:,} wierszy")

    sample_predicate = ""
    if sample_rate and 0 < sample_rate < 1:
        sample_predicate = f" AND random() < {sample_rate:.8f}"

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE events_raw AS
        SELECT
          Timestamp,
          try_cast(LatencyMs AS DOUBLE) AS LatencyMs,
          try_cast(CpuUsage AS DOUBLE) AS CpuUsage,
          try_cast(LocalQps AS DOUBLE) AS LocalQps,
          CAST(IsSpikeByLatency AS BOOLEAN) AS IsSpikeByLatency,
          CAST(IsSpikeByCpu AS BOOLEAN) AS IsSpikeByCpu,
          CAST(IsSpikeByQps AS BOOLEAN) AS IsSpikeByQps
        FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
        WHERE LatencyMs IS NOT NULL AND CpuUsage IS NOT NULL AND LocalQps IS NOT NULL
        {sample_predicate}
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW events AS
        SELECT Timestamp, LatencyMs, CpuUsage, LocalQps
        FROM events_raw
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW events_eval AS
        SELECT
          Timestamp, LatencyMs, CpuUsage, LocalQps,
          IsSpikeByLatency, IsSpikeByCpu, IsSpikeByQps
        FROM events_raw
        """
    )

    print("\n[2/5] Obliczanie statystyk (mean/median/std)...")

    stats = con.execute(
        """
        SELECT
          AVG(LatencyMs) AS lat_mean,
          approx_quantile(LatencyMs, 0.5) AS lat_med,
          stddev_samp(LatencyMs) AS lat_std,

          AVG(CpuUsage) AS cpu_mean,
          approx_quantile(CpuUsage, 0.5) AS cpu_med,
          stddev_samp(CpuUsage) AS cpu_std,

          AVG(LocalQps) AS qps_mean,
          approx_quantile(LocalQps, 0.5) AS qps_med,
          stddev_samp(LocalQps) AS qps_std
        FROM events
        """
    ).fetchone()

    lat_mean, lat_med, lat_std = (float(stats[0]), float(stats[1]), float(stats[2]))
    cpu_mean, cpu_med, cpu_std = (float(stats[3]), float(stats[4]), float(stats[5]))
    qps_mean, qps_med, qps_std = (float(stats[6]), float(stats[7]), float(stats[8]))

    print(f"  LatencyMs: mean={lat_mean:.2f}, median={lat_med:.2f}, std={lat_std:.2f}")
    print(f"  CpuUsage:  mean={cpu_mean:.2f}, median={cpu_med:.2f}, std={cpu_std:.2f}")
    print(f"  LocalQps:  mean={qps_mean:.2f}, median={qps_med:.2f}, std={qps_std:.2f}")

    print("\n[3/5] Metoda progowa - wykrywanie spike'ów (metryki)...")

    lat_3med = 3 * lat_med
    lat_sigma = lat_mean + K_SIGMA * lat_std

    cpu_3med = 3 * cpu_med
    cpu_sigma = cpu_mean + K_SIGMA * cpu_std

    qps_3med = 3 * qps_med
    qps_sigma = qps_mean + K_SIGMA * qps_std

    checks = [
        ("Latency > 3×median", f"LatencyMs > {lat_3med}", "IsSpikeByLatency"),
        ("Latency > mean+3σ", f"LatencyMs > {lat_sigma}", "IsSpikeByLatency"),
        ("CPU > 3×median", f"CpuUsage > {cpu_3med}", "IsSpikeByCpu"),
        ("CPU > mean+3σ", f"CpuUsage > {cpu_sigma}", "IsSpikeByCpu"),
        ("QPS > 3×median", f"LocalQps > {qps_3med}", "IsSpikeByQps"),
        ("QPS > mean+3σ", f"LocalQps > {qps_sigma}", "IsSpikeByQps"),
    ]

    print(f"  {'Reguła':<20} {'Label':<16} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'TP':>10} {'FP':>10} {'FN':>10} {'TN':>10}")
    print("  " + "-" * 97)
    for name, pred, label in checks:
        tp, fp, fn, tn, p, r, f1 = prf1_event_level(con, pred, label)
        print(f"  {name:<20} {label:<16} {p:>7.3f} {r:>7.3f} {f1:>7.3f} "
              f"{tp:>10,} {fp:>10,} {fn:>10,} {tn:>10,}")

    print("\n[4/5] Metoda okien czasowych (QPS) - wykrywanie spike'ów...")

    print(f"  {'Okno':<10} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'TP':>10} {'FP':>10} {'FN':>10} {'TN':>10}")
    print("  " + "-" * 86)
    for w in [1, 5]:
        total, tp, fp, fn, tn, p, r, f1 = prf1_qps_windows(con, w, QPS_WINDOW_MULTIPLIER)
        print(f"  {str(w)+'min':<10} {p:>7.3f} {r:>7.3f} {f1:>7.3f} "
              f"{tp:>10,} {fp:>10,} {fn:>10,} {tn:>10,}")

    print("\n[5/5] Porównanie z IsSpikeBy* wykonane w [3/5] i [4/5] (TP/FP/FN/TN).")

  
    if ENABLE_BONUS_METHOD:
        print("\n" + "=" * 70)
        print("  WŁASNA METODA — ROBUST Z-SCORE (median + MAD) + OKNA")
        print("=" * 70)

        (lat_m, lat_sig_r), (cpu_m, cpu_sig_r), (qps_m, qps_sig_r) = robust_stats_mad(con)

        print("\n[Bonus/1] Robust statystyki (median, robust_sigma=1.4826*MAD):")
        print(f"  Latency: median={lat_m:.2f}, robust_sigma={lat_sig_r:.2f}")
        print(f"  CPU:     median={cpu_m:.2f}, robust_sigma={cpu_sig_r:.2f}")
        print(f"  QPS:     median={qps_m:.2f}, robust_sigma={qps_sig_r:.2f}")

        pred_lat = f"(LatencyMs - {lat_m}) / NULLIF({lat_sig_r}, 0) > {K_LAT_ROBUST}"
        pred_cpu = f"(CpuUsage  - {cpu_m}) / NULLIF({cpu_sig_r}, 0) > {K_CPU_ROBUST}"
        pred_qps = f"(LocalQps  - {qps_m}) / NULLIF({qps_sig_r}, 0) > {K_QPS_ROBUST}"

        print("\n[Bonus/2] Robust progowo (event-level) vs IsSpikeBy* (PRF1 + TN):")
        print(f"{'Reguła (Robust)':<28} {'Label':<16} {'Prec':>7} {'Rec':>7} {'F1':>7} "
              f"{'TP':>10} {'FP':>10} {'FN':>10} {'TN':>10}")
        print("-" * 102)

        for nm, pred, label in [
            (f"Latency robust_z>{K_LAT_ROBUST:g}", pred_lat, "IsSpikeByLatency"),
            (f"CPU robust_z>{K_CPU_ROBUST:g}", pred_cpu, "IsSpikeByCpu"),
            (f"QPS robust_z>{K_QPS_ROBUST:g}", pred_qps, "IsSpikeByQps"),
        ]:
            tp, fp, fn, tn, p, r, f1 = prf1_event_level(con, pred, label)
            print(f"{nm:<28} {label:<16} {p:>7.3f} {r:>7.3f} {f1:>7.3f} "
                  f"{tp:>10,} {fp:>10,} {fn:>10,} {tn:>10,}")

        print("\n[Bonus/3] Robust okienkowo (pred_rate>=threshold) vs IsSpikeBy* (PRF1 + TN na oknach):")
        print(f"  threshold pred_rate >= {ROBUST_WINDOW_PCT_THRESHOLD:.0%}")
        for w in ROBUST_WINDOWS_MINUTES:
            total, tp, fp, fn, tn, p, r, f1 = prf1_robust_windows(
                con, w, pred_qps, "IsSpikeByQps", ROBUST_WINDOW_PCT_THRESHOLD
            )
            print(f"  QPS okno={w}min: Prec={p:.3f} Rec={r:.3f} F1={f1:.3f}  "
                  f"TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,} (total={total:,})")

     
if __name__ == "__main__":
    main()
