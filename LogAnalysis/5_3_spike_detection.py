from __future__ import annotations

from pathlib import Path
import duckdb

THIS_DIR = Path(__file__).resolve().parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"

THREADS = 8
MEMORY_LIMIT_GB = 16

TUNE_PCT = 5.0
EVAL_PCT = 100.0
EVAL_MAX_ROWS = 2_000_000
EVAL_PCT_LARGE = 5.0

K_SIGMA_RANGE = (1.5, 5.0, 0.5)
PCTL_GRID = [0.95, 0.99]
WINDOW_MINUTES = [1, 5]
WINDOW_MULTIPLIER_GRID = [1.5, 2.0, 2.5, 3.0]


def prf1_event_level(
    con: duckdb.DuckDBPyConnection, pred_expr: str, label_col: str, view_name: str
):
    # Klasyczne PRF1 na poziomie eventow
    tp, fp, fn, tn = con.execute(
        f"""
        WITH x AS (
          SELECT
            CASE WHEN {pred_expr} THEN 1 ELSE 0 END AS pred,
            CASE WHEN {label_col} THEN 1 ELSE 0 END AS label
          FROM {view_name}
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


def frange(start: float, stop: float, step: float):
    # Prosty generator zakresu dla k
    k = start
    while k <= stop + 1e-9:
        yield round(k, 3)
        k += step


def best_rule_for_metric(
    con: duckdb.DuckDBPyConnection,
    metric_col: str,
    label_col: str,
    median_val: float,
    mean_val: float,
    std_val: float,
    pctls: dict[float, float],
    tune_view: str,
):
    # Kandydaci reguł progowych tylko z metryki
    candidates = []

    pred_3med = f"{metric_col} > {3 * median_val}"
    candidates.append(("3x median", pred_3med))

    for k in frange(*K_SIGMA_RANGE):
        pred = f"{metric_col} > {mean_val + k * std_val}"
        candidates.append((f"mean+{k:g}σ", pred))

    for p, thr in pctls.items():
        candidates.append((f"p{int(p*100):d}", f"{metric_col} >= {thr}"))

    best = None
    results = []
    for name, pred in candidates:
        tp, fp, fn, tn, p, r, f1 = prf1_event_level(con, pred, label_col, tune_view)
        row = {
            "name": name,
            "pred": pred,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "p": p,
            "r": r,
            "f1": f1,
        }
        results.append(row)
        if best is None or f1 > best["f1"]:
            best = row

    return best, results


def window_grid_results(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    metric_col: str,
    label_col: str,
):
    # Kandydaci reguł okienkowych (porównanie okien czasowych)
    results = []
    for w in WINDOW_MINUTES:
        for mult in WINDOW_MULTIPLIER_GRID:
            tp, fp, fn, tn, p, r, f1 = prf1_metric_windows(
                con, metric_col, label_col, w, mult, view_name
            )
            results.append(
                {
                    "name": f"window {w}m x{mult:g}",
                    "pred": None,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "p": p,
                    "r": r,
                    "f1": f1,
                    "window": (w, mult),
                }
            )
    return results


def best_of(a, b):
    # Wybór najlepszego po F1
    best = a
    for row in b:
        if row["f1"] > best["f1"]:
            best = row
    return best


def prf1_metric_windows(
    con: duckdb.DuckDBPyConnection,
    metric_col: str,
    label_col: str,
    window_minutes: int,
    multiplier: float,
    view_name: str,
):
    # PRF1 na poziomie okien (avg w oknie vs poprzednie okno)
    row = con.execute(
        f"""
        WITH win AS (
          SELECT
            date_trunc('minute', Timestamp)
              - (EXTRACT(minute FROM Timestamp)::INTEGER % {window_minutes}) * INTERVAL '1 minute' AS win_start,
            AVG({metric_col}) AS avg_val,
            MAX(CASE WHEN {label_col} THEN 1 ELSE 0 END) AS label
          FROM {view_name}
          GROUP BY 1
        ),
        lagged AS (
          SELECT
            win_start,
            avg_val,
            label,
            LAG(avg_val) OVER (ORDER BY win_start) AS prev_val
          FROM win
        ),
        scored AS (
          SELECT
            CASE WHEN prev_val IS NOT NULL AND avg_val > {multiplier} * prev_val THEN 1 ELSE 0 END AS pred,
            label
          FROM lagged
          WHERE prev_val IS NOT NULL
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
    return tp, fp, fn, tn, precision, recall, f1




def main():
    if not PARQUET_DIR.exists():
        raise RuntimeError(f"Brak danych: {PARQUET_DIR}")

    print("=" * 70)
    print("  5.3 WYKRYWANIE SPIKE'OW (bez flag IsSpike* i SpikeWindow)")
    print("=" * 70)

    # DuckDB na parquetach jest szybkie i skaluje się na duże dane
    con = duckdb.connect()
    con.execute("PRAGMA threads=?", [THREADS])
    con.execute("PRAGMA memory_limit=?", [f"{MEMORY_LIMIT_GB}GB"])

    print("\n[1/4] Wczytywanie danych...")
    # Liczba wierszy -> decyzja czy ewaluacja na całych danych czy próbce
    total_rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
        WHERE LatencyMs IS NOT NULL AND CpuUsage IS NOT NULL AND LocalQps IS NOT NULL
        """
    ).fetchone()[0]

    tune_rate = None
    if TUNE_PCT and 0 < TUNE_PCT <= 100:
        tune_rate = TUNE_PCT / 100.0

    eval_rate = None
    eval_pct_effective = EVAL_PCT
    if total_rows > EVAL_MAX_ROWS:
        eval_pct_effective = EVAL_PCT_LARGE
    if eval_pct_effective and 0 < eval_pct_effective <= 100:
        eval_rate = eval_pct_effective / 100.0
        label = "wszystkie dane" if eval_pct_effective == 100 else f"{eval_pct_effective:.2f}%"

    tune_predicate = ""
    if tune_rate and 0 < tune_rate < 1:
        tune_predicate = f" AND random() < {tune_rate:.8f}"

    eval_predicate = ""
    if eval_rate and 0 < eval_rate < 1:
        eval_predicate = f" AND random() < {eval_rate:.8f}"

    # Próbka do tuningu (dobór najlepszej reguły)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE events_tune AS
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
        {tune_predicate}
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW events_tune_v AS
        SELECT * FROM events_tune
        """
    )


    print("\n[2/4] Statystyki...")
    # Statystyki dla progów (mean/median/std + percentyle)
    stats = con.execute(
        """
        WITH base AS (
          SELECT
            AVG(LatencyMs) AS lat_mean,
            approx_quantile(LatencyMs, 0.5) AS lat_med,
            stddev_samp(LatencyMs) AS lat_std,
            approx_quantile(LatencyMs, 0.95) AS lat_p95,
            approx_quantile(LatencyMs, 0.99) AS lat_p99,
            AVG(CpuUsage) AS cpu_mean,
            approx_quantile(CpuUsage, 0.5) AS cpu_med,
            stddev_samp(CpuUsage) AS cpu_std,
            approx_quantile(CpuUsage, 0.95) AS cpu_p95,
            approx_quantile(CpuUsage, 0.99) AS cpu_p99,
            AVG(LocalQps) AS qps_mean,
            approx_quantile(LocalQps, 0.5) AS qps_med,
            stddev_samp(LocalQps) AS qps_std,
            approx_quantile(LocalQps, 0.95) AS qps_p95,
            approx_quantile(LocalQps, 0.99) AS qps_p99
          FROM events_tune_v
        )
        SELECT
          lat_mean, lat_med, lat_std,
          lat_p95, lat_p99,
          cpu_mean, cpu_med, cpu_std,
          cpu_p95, cpu_p99,
          qps_mean, qps_med, qps_std,
          qps_p95, qps_p99
        FROM base
        """
    ).fetchone()

    (
        lat_mean, lat_med, lat_std,
        lat_p95, lat_p99,
        cpu_mean, cpu_med, cpu_std,
        cpu_p95, cpu_p99,
        qps_mean, qps_med, qps_std,
        qps_p95, qps_p99,
    ) = map(float, stats)

    print(f"  Latency: mean={lat_mean:.2f}, median={lat_med:.2f}, std={lat_std:.2f}")
    print(f"  CPU:     mean={cpu_mean:.2f}, median={cpu_med:.2f}, std={cpu_std:.2f}")
    print(f"  QPS:     mean={qps_mean:.2f}, median={qps_med:.2f}, std={qps_std:.2f}")

    # Wspólna ścieżka dla Latency/CPU/QPS
    metrics = [
        ("Latency", "LatencyMs", "IsSpikeByLatency", lat_med, lat_mean, lat_std, lat_p95, lat_p99),
        ("CPU", "CpuUsage", "IsSpikeByCpu", cpu_med, cpu_mean, cpu_std, cpu_p95, cpu_p99),
        ("QPS", "LocalQps", "IsSpikeByQps", qps_med, qps_mean, qps_std, qps_p95, qps_p99),
    ]
    best_map = {}
    for name, col, label, med, mean, std, p95, p99 in metrics:
        pctls = {0.95: p95, 0.99: p99}
        best_rule, _ = best_rule_for_metric(
            con, col, label, med, mean, std, pctls, "events_tune_v"
        )
        win = window_grid_results(con, "events_tune_v", col, label)
        best_map[name] = best_of(best_rule, win)

    # Zbiór do ewaluacji jakości wybranych reguł
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE events_eval AS
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
        {eval_predicate}
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW events_eval_v AS
        SELECT * FROM events_eval
        """
    )

    # Końcowy raport PRF1
    print("\n[4/4] Raport PRF1 dla najlepszych reguł na zbiorze ewaluacyjnym:")
    print(f"{'Metryka':<10} {'Reguła':<14} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TP':>10} {'FP':>10} {'FN':>10} {'TN':>10}")
    print("-" * 86)
    for name, col, label, *_ in metrics:
        best = best_map[name]
        if best.get("pred"):
            tp, fp, fn, tn, p, r, f1 = prf1_event_level(
                con, best["pred"], label, "events_eval_v"
            )
        else:
            w, mult = best["window"]
            tp, fp, fn, tn, p, r, f1 = prf1_metric_windows(
                con, col, label, w, mult, "events_eval_v"
            )
        print(
            f"{name:<10} {best['name']:<14} {p:>7.3f} {r:>7.3f} {f1:>7.3f} "
            f"{tp:>10,} {fp:>10,} {fn:>10,} {tn:>10,}"
        )



if __name__ == "__main__":
    main()
