"""
5.4 Trend detection (DuckDB + Python)

Założenia:
- okno czasowe: 5 minut
- trend: min. 1 godzina (12 okien po 5 min)
- próg nachylenia: +2 ms / okno 5-min => 0.4 ms / minutę
- jakość dopasowania: R^2 >= 0.4
- TrendWindow używane tylko do walidacji (nie do wykrywania)
"""

from __future__ import annotations

from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ŚCIEŻKI

THIS_DIR = Path(__file__).resolve().parent           # LogAnalysis/
REPO_ROOT = THIS_DIR.parent                          # root projektu

PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"     # LogAnalysis/logs_expanded_parquet
STATS_JSON = REPO_ROOT / "logs.csv.stats.json"       # root/logs.csv.stats.json

# PARAMETRY

SOURCE_SYSTEM = "OrderService"
METRIC = "LatencyMs"

WINDOW_MINUTES = 5
SEGMENT_WINDOWS = 12         # 12 * 5 min = 1 godzina

# Regresja "czas → wartość" w minutach:
# +2 ms / okno 5-min = +0.4 ms/min
SLOPE_THRESHOLD_MS_PER_MIN = 0.4
MIN_R2 = 0.4

# zasoby DuckDB
THREADS = 8
MEMORY_LIMIT_GB = 16
TEMP_DIR = "/tmp/duckdb"


# 1) WCZYTANIE + OKNA CZASOWE

if not PARQUET_DIR.exists():
    raise RuntimeError(f"Nie znaleziono katalogu parquet: {PARQUET_DIR}")

con = duckdb.connect()
con.execute("PRAGMA threads=?", [THREADS])
con.execute("PRAGMA memory_limit=?", [f"{MEMORY_LIMIT_GB}GB"])
con.execute("PRAGMA temp_directory=?", [TEMP_DIR])

# Jedno zapytanie:
# - agreguje do okien 5-min
# - liczy średnią metryki
# - liczy ile w oknie było TrendWindow=true (tylko walidacja)
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
    AVG(value) AS avg_value,
    COUNT(*) AS n_events,
    SUM(CASE WHEN TrendWindow THEN 1 ELSE 0 END) AS n_trendwindow
FROM base
GROUP BY time_window
ORDER BY time_window
"""

df = con.execute(sql, [SOURCE_SYSTEM]).df()

print(f"\n[INFO] System={SOURCE_SYSTEM}, Metric={METRIC}")
print(f"[INFO] ParquetDir={PARQUET_DIR}")
print(f"[INFO] Liczba okien 5-min: {len(df)}")
print(df.head(10).to_string(index=False))

if df.empty or len(df) < SEGMENT_WINDOWS:
    raise RuntimeError("Za mało danych/okien do wykrycia trendu (sprawdź SourceSystem / metrykę).")

df["time_window"] = pd.to_datetime(df["time_window"])


# 2) REGRESJA LINIOWA: CZAS -> WARTOŚĆ

def linear_regression_time(time_windows: pd.Series, values: np.ndarray) -> tuple[float, float]:
    """
    Dopasowuje prostą: avg_value ~ czas (w minutach)
    Zwraca:
      slope_ms_per_min (ms/min)
      r2
    """
    t0 = time_windows.iloc[0]
    # czas w minutach od początku segmentu
    X = ((time_windows - t0).dt.total_seconds() / 60.0).to_numpy().reshape(-1, 1)
    y = values.astype(float)

    model = LinearRegression()
    model.fit(X, y)

    slope_ms_per_min = float(model.coef_[0])
    r2 = float(model.score(X, y))
    return slope_ms_per_min, r2


# 3) WYKRYWANIE ODCINKÓW TRENDU (przesuwane okno 1h)

avg_vals = df["avg_value"].to_numpy(dtype=float)
tw_ratio_per_window = (df["n_trendwindow"] / df["n_events"]).fillna(0.0).to_numpy(dtype=float)

trend_segments = []

for i in range(len(df) - SEGMENT_WINDOWS + 1):
    seg_times = df["time_window"].iloc[i : i + SEGMENT_WINDOWS]
    seg_vals = avg_vals[i : i + SEGMENT_WINDOWS]

    slope_ms_per_min, r2 = linear_regression_time(seg_times, seg_vals)

    if slope_ms_per_min >= SLOPE_THRESHOLD_MS_PER_MIN and r2 >= MIN_R2:
        trend_segments.append({
            "start": seg_times.iloc[0],
            "end": seg_times.iloc[-1],
            "slope_ms_per_min": slope_ms_per_min,
            "r2": r2,
            # walidacja: średni udział TrendWindow w oknach tego segmentu
            "trendwindow_ratio_avg": float(np.mean(tw_ratio_per_window[i : i + SEGMENT_WINDOWS])),
        })

trend_df = pd.DataFrame(trend_segments)

print("\n[WYNIK] Wykryte odcinki trendu (przed scalaniem):")
if trend_df.empty:
    print("Brak wykrytych trendów.")
    print("TIP: jeśli wiesz, że trend jest, obniż próg SLOPE_THRESHOLD_MS_PER_MIN do 0.2 albo MIN_R2 do 0.3.")
    raise SystemExit(0)
else:
    print(trend_df.to_string(index=False))


# 4) SCALANIE NACHODZĄCYCH ODCINKÓW

trend_df = trend_df.sort_values(["start", "end"]).reset_index(drop=True)
merged = []
cur = trend_df.iloc[0].to_dict()

for j in range(1, len(trend_df)):
    row = trend_df.iloc[j]
    if row["start"] <= cur["end"]:
        cur["end"] = max(cur["end"], row["end"])
        cur["slope_ms_per_min"] = max(cur["slope_ms_per_min"], float(row["slope_ms_per_min"]))
        cur["r2"] = max(cur["r2"], float(row["r2"]))
        cur["trendwindow_ratio_avg"] = max(cur["trendwindow_ratio_avg"], float(row["trendwindow_ratio_avg"]))
    else:
        merged.append(cur)
        cur = row.to_dict()
merged.append(cur)

merged_df = pd.DataFrame(merged)

print("\n[WYNIK KOŃCOWY] Wykryte odcinki trendu (po scaleniu):")
print(merged_df.to_string(index=False))


# 5) LICZBA EVENTÓW TRENDU

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

trend_events_sql = f"""
SELECT COUNT(*) AS trend_events
FROM read_parquet(
    '{str(PARQUET_DIR).replace("'", "''")}/**/*.parquet',
    hive_partitioning=true
)
WHERE SourceSystem = '{SOURCE_SYSTEM}'
  AND EXISTS (
      SELECT 1
      FROM trend_segments s
      WHERE Timestamp BETWEEN s.start_ts AND s.end_ts
  )
"""

trend_events = int(con.execute(trend_events_sql).fetchone()[0])
print(f'\n"TrendEvents" (nasza metoda): {trend_events}')


# 6) WALIDACJA: ile eventów ma TrendWindow=true (tylko OrderService)

trend_events_tw_sql = f"""
SELECT COUNT(*) AS trend_events_tw
FROM read_parquet(
    '{str(PARQUET_DIR).replace("'", "''")}/**/*.parquet',
    hive_partitioning=true
)
WHERE SourceSystem = '{SOURCE_SYSTEM}'
  AND TrendWindow = true
"""
trend_events_tw = int(con.execute(trend_events_tw_sql).fetchone()[0])
print(f'"TrendEvents_TrendWindow" (OrderService): {trend_events_tw}')


# 7) PORÓWNANIE DO logs.csv.stats.json (dla wszystkich systemów)

if STATS_JSON.exists():
    stats = json.loads(STATS_JSON.read_text(encoding="utf-8"))
    print(f'\n[STATS.JSON] znaleziono: {STATS_JSON}')
    print(f'[STATS.JSON] TrendEvents (global): {stats.get("TrendEvents")}')
else:
    print(f"\n[STATS.JSON] Nie znaleziono pliku: {STATS_JSON}")


# Nasza metoda wykrywa ~57% eventów trendowych oznaczonych przez generator dla OrderService.
