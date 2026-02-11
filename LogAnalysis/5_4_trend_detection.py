"""
5.4 Trend detection (DuckDB + Python) — variable-length trends, anti-spike
"""

from __future__ import annotations
from pathlib import Path
import json
import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# =============================================================================
# 0. PARAMETRY I ŚCIEŻKI
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"
STATS_JSON = REPO_ROOT / "logs.csv.stats.json"

SOURCE_SYSTEM = "OrderService"
METRIC = "LatencyMs"

WINDOW_MINUTES = 5

MIN_R2 = 0.4 # współczynnik determinacji
MIN_EVENTS_PER_WINDOW_WARN = 5

# Trend logic
MIN_SEG_WINDOWS = 6             # min 30 minut
MAX_BAD_DIFF_RATIO = 0.10        # ~90% przyrostów dodatnich
SPIKE_STD_FACTOR = 3  # ile odchyleń standardowych uznajemy za spike

# Adaptive slope
ADAPTIVE_SLOPE_PERCENTILE = 75
ADAPTIVE_SLOPE_MIN_FLOOR = 0.2 # milisekundy na minutę (ms/min)
ADAPTIVE_SLOPE_MAX_CAP = 2.0

THREADS = 8
MEMORY_LIMIT_GB = 16
TEMP_DIR = "/tmp/duckdb"

# =============================================================================
# 1. WCZYTANIE I AGREGACJA DO OKIEN 5-MIN
# =============================================================================

print("\n=== [1] WCZYTANIE I AGREGACJA DANYCH ===")

if not PARQUET_DIR.exists():
    raise RuntimeError(f"Nie znaleziono katalogu parquet: {PARQUET_DIR}")

con = duckdb.connect()
con.execute("PRAGMA threads=?", [THREADS])
con.execute("PRAGMA memory_limit=?", [f"{MEMORY_LIMIT_GB}GB"])
con.execute("PRAGMA temp_directory=?", [TEMP_DIR])

sql = f"""
WITH base AS (
    SELECT
        Timestamp,
        {METRIC} AS value,
        TrendWindow
    FROM read_parquet(
        '{str(PARQUET_DIR).replace("'", "''")}/**/*.parquet',
        hive_partitioning=true
    )
    WHERE SourceSystem = ?
      AND {METRIC} IS NOT NULL
)
SELECT
    date_trunc('minute', Timestamp)
      - (EXTRACT(minute FROM Timestamp)::INT % {WINDOW_MINUTES}) * INTERVAL '1 minute'
        AS time_window,
    median(value) AS med_value,
    COUNT(*) AS n_events,
    SUM(CASE WHEN TrendWindow THEN 1 ELSE 0 END) AS n_trendwindow
FROM base
GROUP BY time_window
ORDER BY time_window
"""

df = con.execute(sql, [SOURCE_SYSTEM]).df()
df["time_window"] = pd.to_datetime(df["time_window"])

print(f"System: {SOURCE_SYSTEM}, Metryka: {METRIC}")
print(f"Liczba okien 5-minutowych: {len(df)}")
print(df.head(5).to_string(index=False))

if df.empty:
    raise RuntimeError("Brak danych po filtrze.")

# Uzupełnienie osi czasu 
df = df.sort_values("time_window").set_index("time_window")
df = df.asfreq(f"{WINDOW_MINUTES}min")
df["n_events"] = df["n_events"].fillna(0).astype(int)
df["n_trendwindow"] = df["n_trendwindow"].fillna(0).astype(int)
df = df.reset_index()

median_events = float(np.median(df["n_events"].to_numpy()))
if median_events < MIN_EVENTS_PER_WINDOW_WARN:
    print(f"[WARN] Mało eventów na okno: mediana={median_events:.1f}")

# =============================================================================
# 2. REGRESJA LINIOWA
# =============================================================================

def linear_regression_time(time_windows: pd.Series, values: np.ndarray) -> tuple[float, float]:
    t0 = time_windows.iloc[0]
    X = ((time_windows - t0).dt.total_seconds() / 60.0).to_numpy().reshape(-1, 1)
    y = values.astype(float)
    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0]), float(model.score(X, y)) # slope, r2

# =============================================================================
# 3. ADAPTACYJNY PRÓG SLOPE 
# =============================================================================

print("\n=== [2] ADAPTACYJNY PRÓG SLOPE ===")

valid = df[df["n_events"] > 0].copy().reset_index(drop=True)
times = valid["time_window"]
vals = valid["med_value"].to_numpy(dtype=float)

rng = np.random.default_rng(42)
all_slopes = []

for _ in range(min(2000, len(valid) * 5)):
    w = rng.integers(3, min(24, len(valid)))
    i = rng.integers(0, len(valid) - w + 1)
    slope, _ = linear_regression_time(times.iloc[i:i+w], vals[i:i+w])
    all_slopes.append(slope)

adaptive_thr = float(np.percentile(all_slopes, ADAPTIVE_SLOPE_PERCENTILE))
adaptive_thr = max(adaptive_thr, ADAPTIVE_SLOPE_MIN_FLOOR)
adaptive_thr = min(adaptive_thr, ADAPTIVE_SLOPE_MAX_CAP)

print(f"Adaptacyjny próg slope: {adaptive_thr:.3f} ms/min")

# =============================================================================
# 4. DETEKCJA ODCINKÓW TRENDU 
# =============================================================================

print("\n=== [3] DETEKCJA ODCINKÓW TRENDU ===")

diffs = np.diff(vals)
mean_diff = np.mean(diffs)
std_diff = np.std(diffs)
spike_thr_upper = mean_diff + SPIKE_STD_FACTOR * std_diff
spike_thr_lower = mean_diff - SPIKE_STD_FACTOR * std_diff

trend_segments = []
start = 0
bad = 0
count = 0

for i in range(len(diffs)):
    count += 1

    if diffs[i] <= 0:
        bad += 1

    # spike
    if diffs[i] > spike_thr_upper or diffs[i] < spike_thr_lower:
        bad = count

    if (bad / count) > MAX_BAD_DIFF_RATIO:
        end = i
        if (end - start + 1) >= MIN_SEG_WINDOWS:
            seg_times = times.iloc[start:end+1]
            seg_vals = vals[start:end+1]
            slope, r2 = linear_regression_time(seg_times, seg_vals)

            if slope >= adaptive_thr and r2 >= MIN_R2:
                trend_segments.append({
                    "start": seg_times.iloc[0],
                    "end": seg_times.iloc[-1],
                    "slope_ms_per_min": slope,
                    "r2": r2,
                })

        start = i + 1
        bad = 0
        count = 0

# domknięcie końcówki
end = len(vals) - 1
if (end - start + 1) >= MIN_SEG_WINDOWS:
    seg_times = times.iloc[start:end+1]
    seg_vals = vals[start:end+1]
    slope, r2 = linear_regression_time(seg_times, seg_vals)

    if slope >= adaptive_thr and r2 >= MIN_R2:
        trend_segments.append({
            "start": seg_times.iloc[0],
            "end": seg_times.iloc[-1],
            "slope_ms_per_min": slope,
            "r2": r2,
        })

trend_df = pd.DataFrame(trend_segments)
print(f"Wykryto {len(trend_df)} segmentów trendu.")
if trend_df.empty:
    print("Brak trendów.")
    raise SystemExit(0)
print(trend_df.head(5).to_string(index=False))

# =============================================================================
# 5. SCALANIE SEGMENTÓW
# =============================================================================

print("\n=== [4] SCALANIE SEGMENTÓW ===")

MERGE_GAP = pd.Timedelta(minutes=WINDOW_MINUTES)

trend_df = trend_df.sort_values(["start", "end"]).reset_index(drop=True)
merged = []
cur = trend_df.iloc[0].to_dict()

for i in range(1, len(trend_df)):
    row = trend_df.iloc[i].to_dict()
    if row["start"] <= cur["end"] + MERGE_GAP:
        cur["end"] = max(cur["end"], row["end"])
        cur["slope_ms_per_min"] = max(cur["slope_ms_per_min"], row["slope_ms_per_min"])
        cur["r2"] = max(cur["r2"], row["r2"])
    else:
        merged.append(cur)
        cur = row

merged.append(cur)
merged_df = pd.DataFrame(merged)
print(f"Po scaleniu: {len(merged_df)} segmentów.")
print(merged_df.head(5).to_string(index=False))

# =============================================================================
# 6. WALIDACJA (TrendWindow)
# =============================================================================

print("\n=== [5] WALIDACJA I METRYKI ===")

con.execute("DROP TABLE IF EXISTS trend_segments")
con.execute("""
CREATE TEMP TABLE trend_segments (
    start_ts TIMESTAMP,
    end_ts   TIMESTAMP
)
""")

con.executemany(
    "INSERT INTO trend_segments VALUES (?, ?)",
    list(merged_df[["start", "end"]].itertuples(index=False, name=None))
)

confusion_sql = f"""
WITH events AS (
    SELECT
        Timestamp,
        TrendWindow AS truth,
        EXISTS (
            SELECT 1 FROM trend_segments s
            WHERE Timestamp BETWEEN s.start_ts AND s.end_ts
        ) AS pred
    FROM read_parquet(
        '{str(PARQUET_DIR).replace("'", "''")}/**/*.parquet',
        hive_partitioning=true
    )
    WHERE SourceSystem = ?
)
SELECT
    SUM(CASE WHEN pred AND truth THEN 1 ELSE 0 END) AS tp,
    SUM(CASE WHEN pred AND NOT truth THEN 1 ELSE 0 END) AS fp,
    SUM(CASE WHEN NOT pred AND truth THEN 1 ELSE 0 END) AS fn,
    SUM(CASE WHEN NOT pred AND NOT truth THEN 1 ELSE 0 END) AS tn
FROM events
"""
# TP— algorytm trafnie wykrył trend 
# FP (False Positive) — algorytm błędnie wykrył trend 
# FN (False Negative) — algorytm nie znalazł trendu 
# TN (True Negative) — algorytm trafnie odrzucił trend 

tp, fp, fn, tn = map(int, con.execute(confusion_sql, [SOURCE_SYSTEM]).fetchone())
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
fpr = fp / (fp + tn) if (fp + tn) else 0.0

print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f} FPR={fpr:.4f}")

if STATS_JSON.exists():
    stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
    print(f"[STATS.JSON] TrendEvents (global): {stats.get('TrendEvents')}")
