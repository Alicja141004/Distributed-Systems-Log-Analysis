import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path
import sys
import time

THIS_DIR = Path(__file__).resolve().parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"
SAMPLE_SIZE = 500000 

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', 50)

def interpret_cluster_logic(row):
    """
    Logika biznesowa interpretacji klastra na podstawie średnich wartości cech.
    Kolejność warunków ma znaczenie!
    """
    lat = row['Avg_Lat']
    cpu = row['Avg_CPU']
    err = row['Avg_NetErr']
    qps = row['Avg_QPS']
    mem = row['Avg_Mem']
    
    # 1. ANOMALY (Extr. Outliers) - sprawdzamy najpierw, żeby nie wpadło do Failure
    # Event 999 z generatora ma np. Latency > 2000-10000ms
    if lat > 4000:
        return "ANOMALY (Extr. Latency)"
    
    # 2. CRITICAL FAILURE (System Failure)
    # Event 500/501 - Latency boost 500-2000, Errors
    if lat > 1000 or err > 15.0:
        return "CRITICAL FAILURE"
        
    # 3. SPIKE / HEAVY LOAD
    # CPU > 80% lub QPS > 100 (generator settings)
    if cpu > 75.0 or qps > 150.0:
        if lat > 300:
            return "HEAVY LOAD (Spike+Lag)"
        else:
            return "HEAVY TRAFFIC (Processing)"

    # 4. ELEVATED LOAD
    if cpu > 50.0 or lat > 100:
        return "ELEVATED LOAD"
        
    # 5. NORMAL
    return "NORMAL"

def analyze_and_print_results(df, k):
    print(f"\n{'#'*160}")
    print(f" WYNIKI K-MEANS (K={k})")
    print(f"{'#'*160}")

    # Agregacja danych dla każdego klastra
    summary = df.groupby('Cluster').agg(
        # --- CECHY TRENINGOWE (8 wymiarów) ---
        Avg_Lat=('LatencyMs', 'mean'),
        Avg_CPU=('CpuUsage', 'mean'),
        Avg_Mem=('MemoryUsageMb', 'mean'),
        Avg_Disk=('DiskQueueLength', 'mean'),
        Avg_NetErr=('NetworkErrors', 'mean'),
        Avg_QPS=('LocalQps', 'mean'),
        Avg_Req=('RequestSizeBytes', 'mean'),
        Avg_Resp=('ResponseSizeBytes', 'mean'),
        
        # --- WERYFIKACJA (Ground Truth Comparison) ---
        # 1. Anomalie i Błędy
        IsAnom_Pct=('IsAnomaly', lambda x: np.mean(x) * 100),
        
        # 2. Priority Distribution (Wymagane przez Ciebie)
        Warn_Pct=('Priority', lambda x: (x == 'warn').mean() * 100),
        Err_Pct=('Priority', lambda x: (x == 'err').mean() * 100),
        Crit_Pct=('Priority', lambda x: (x == 'crit').mean() * 100),
        
        # 3. Spike Types
        SpikeLat_Pct=('IsSpikeByLatency', lambda x: np.mean(x) * 100),
        SpikeCpu_Pct=('IsSpikeByCpu', lambda x: np.mean(x) * 100),
        
        # 4. Metadata
        Top_Event=('EventCode', lambda x: x.mode()[0] if not x.mode().empty else 0),
        Count=('LatencyMs', 'count')
    ).reset_index()

    # Interpretacja
    summary['Interpretation'] = summary.apply(interpret_cluster_logic, axis=1)

    # Formatowanie tabeli
    print(f"{'ID':<3} | {'INTERPRETACJA':<23} | {'Count':<7} || "
          f"{'Lat':<6} | {'CPU':<4} | {'Mem':<5} | {'Disk':<4} | {'NetE':<4} | {'QPS':<5} | {'ReqS':<5} | {'RspS':<5} || "
          f"{'Anom%':<5} | {'Warn%':<5} | {'Err%':<4} | {'Crit%':<5} || "
          f"{'SpikeL%':<7} | {'SpikeC%':<7} | {'Event':<5}")
    print("-" * 160)
    
    for _, row in summary.iterrows():
        print(f"{int(row['Cluster']):<3} | {row['Interpretation']:<23} | {int(row['Count']):<7} || "
              f"{row['Avg_Lat']:<6.0f} | {row['Avg_CPU']:<4.0f} | {row['Avg_Mem']:<5.0f} | {row['Avg_Disk']:<4.1f} | {row['Avg_NetErr']:<4.1f} | {row['Avg_QPS']:<5.0f} | {row['Avg_Req']/1000:<5.1f} | {row['Avg_Resp']/1000:<5.1f} || "
              f"{row['IsAnom_Pct']:<5.1f} | {row['Warn_Pct']:<5.1f} | {row['Err_Pct']:<4.1f} | {row['Crit_Pct']:<5.1f} || "
              f"{row['SpikeLat_Pct']:<7.1f} | {row['SpikeCpu_Pct']:<7.1f} | {int(row['Top_Event']):<5}")
    
    print("-" * 160)
    print("LEGENDA KOLUMN:")
    print("  Feature Vector (Avg): Lat(ms), CPU(%), Mem(MB), Disk(len), NetE(count), QPS, ReqS(kB), RspS(kB)")
    print("  Ground Truth (%):     Anom=IsAnomaly, Warn/Err/Crit=Priority, SpikeL=IsSpikeByLatency, SpikeC=IsSpikeByCpu")

def run_clustering():
    if not PARQUET_DIR.exists():
        print(f"BŁĄD: Nie znaleziono {PARQUET_DIR}")
        sys.exit(1)
        
    start_time = time.time()
    con = duckdb.connect()
    parquet_path_str = str(PARQUET_DIR).replace('\\', '/')
    
    print(f"Start K-Means (5.9).")
    
    # 1. Feature Vector (8 cech)
    feature_cols = [
        'LatencyMs', 'CpuUsage', 'MemoryUsageMb', 'DiskQueueLength', 
        'NetworkErrors', 'LocalQps', 'RequestSizeBytes', 'ResponseSizeBytes'
    ]
    
    # 2. Ground Truth Columns (do weryfikacji)
    meta_cols = [
        'Priority', 'EventCode', 'IsAnomaly', 
        'IsSpikeByLatency', 'IsSpikeByCpu', 'IsSpikeByQps'
    ]
    
    select_cols = ", ".join(feature_cols + meta_cols)
    
    print(f"Pobieranie danych (SAMPLE={SAMPLE_SIZE})...")
    
    query = f"""
    SELECT {select_cols}
    FROM read_parquet('{parquet_path_str}/**/*.parquet', hive_partitioning=true)
    WHERE LatencyMs IS NOT NULL AND CpuUsage IS NOT NULL
    USING SAMPLE {SAMPLE_SIZE}
    """
    
    try:
        df = con.execute(query).df()
    except Exception as e:
        print(f"Błąd pobierania danych: {e}")
        return

    print(f"Pobrano {len(df)} wierszy.")
    print(f"Trenowanie na pełnym wektorze cech: {feature_cols}")

    # 3. Preprocessing
    X = df[feature_cols].copy()
    X = X.fillna(0) 
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. K-Means Loop
    k_values = [3, 5, 8]
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        analyze_and_print_results(df, k)

    print(f"\nAnaliza zakończona w: {time.time() - start_time:.2f} s")

if __name__ == "__main__":
    run_clustering()