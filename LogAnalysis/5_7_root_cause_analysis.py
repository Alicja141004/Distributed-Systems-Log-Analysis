from pathlib import Path
import duckdb
import pandas as pd

# ŚCIEŻKI

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"


# 1. WYBÓR CorrelationId Z FAILURE

con = duckdb.connect()

cid_sql = f"""
SELECT CorrelationId
FROM read_parquet('{str(PARQUET_DIR).replace("'", "''")}/**/*.parquet', hive_partitioning=true)
WHERE EventCode IN (500, 501, 998, 999)
    AND ScenarioStepIndex IS NOT NULL            -- wywala heartbeat/metryki z <NA>
GROUP BY CorrelationId
HAVING COUNT(*) >= 8                           -- ma mieć historię (wiele eventów)
     AND COUNT(DISTINCT SourceSystem) >= 3       -- ma przejść przez wiele usług
ORDER BY COUNT(*) DESC
LIMIT 1;

"""
CORRELATION_ID = con.execute(cid_sql).fetchone()[0]

print(f"[INFO] Wybrany CorrelationId: {CORRELATION_ID}")


# 2. WCZYTANIE HISTORII

sql = f"""
SELECT
    Timestamp,
    ScenarioStepIndex,
    SourceSystem,
    EventCode,
    Priority,
    LatencyMs,
    CpuUsage,
    DiskQueueLength,
    NetworkErrors,
    Retries,
    Description
FROM read_parquet(
    '{str(PARQUET_DIR).replace("'", "''")}/**/*.parquet',
    hive_partitioning=true
)
WHERE CorrelationId = ?
ORDER BY Timestamp, ScenarioStepIndex
"""

df = con.execute(sql, [CORRELATION_ID]).df()

print("\n[HISTORIA TRANSAKCJI]")
print(df.to_string(index=False))


# 3. ODTWORZENIE ŚCIEŻKI SYSTEMÓW

path = df["SourceSystem"].tolist()
path_unique = list(dict.fromkeys(path))

print("\n[ŚCIEŻKA PRZEJŚCIA SYSTEMÓW]")
print(" → ".join(path_unique))


# 4. IDENTYFIKACJA ROOT CAUSE

# pierwszy system z wyraźnym pogorszeniem
df["badness"] = (
    (df["LatencyMs"] > df["LatencyMs"].median() * 2).astype(int)
    + (df["EventCode"].isin([500, 501, 999, 998])).astype(int)
    + (df["Retries"] > 0).astype(int)
    + (df["NetworkErrors"] > 0).astype(int)
)

root_event = df.sort_values("Timestamp").iloc[df["badness"].idxmax()]

print("\n[ROOT CAUSE EVENT]")
print(root_event)


# 5. SYSTEMY DOTKNIĘTE PROBLEMEM

affected = (
    df.groupby("SourceSystem")
      .agg(
          max_latency=("LatencyMs", "max"),
          errors=("EventCode", lambda x: x.isin([500, 501, 999, 998]).sum()),
          retries=("Retries", "sum")
      )
      .reset_index()
)

print("\n[SYSTEMY DOTKNIĘTE PROBLEMEM]")
print(affected.to_string(index=False))

# 6. CO DZIAŁO SIĘ TUŻ PRZED AWARIĄ (LOOK-BACK)

root_ts = pd.to_datetime(root_event["Timestamp"])
root_system = root_event["SourceSystem"]

LOOKBACK_MINUTES = 5

print("\n[LOOK-BACK] Analiza zdarzeń PRZED awarią")
print(f"System: {root_system}")
print(f"Zakres: {LOOKBACK_MINUTES} minut przed {root_ts}")

lookback_sql = f"""
SELECT
    Timestamp,
    EventCode,
    Priority,
    LatencyMs,
    CpuUsage,
    DiskQueueLength,
    NetworkErrors,
    Retries,
    Description
FROM read_parquet(
    '{str(PARQUET_DIR).replace("'", "''")}/**/*.parquet',
    hive_partitioning=true
)
WHERE SourceSystem = ?
  AND Timestamp BETWEEN
        ? - INTERVAL '{LOOKBACK_MINUTES} minutes'
    AND ?
ORDER BY Timestamp DESC
LIMIT 30
"""


before_df = con.execute(
    lookback_sql,
    [root_system, root_ts, root_ts]
).df()

if before_df.empty:
    print("Brak wcześniejszych zdarzeń w tym oknie czasowym.")
else:
    print("\n[ZDARZENIA PRZED AWARIĄ]")
    print(before_df.to_string(index=False))

