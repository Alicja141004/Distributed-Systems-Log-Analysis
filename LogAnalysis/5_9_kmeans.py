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

def interpret_cluster_dynamic(row, global_stats):
    c_lat = row['Avg_Lat']
    c_cpu = row['Avg_CPU']
    c_err = row['Avg_NetErr']
    c_qps = row['Avg_QPS']
    
    # 1. ANOMALY (Extreme)
    if c_lat > global_stats['Lat_p99']:
        return "ANOMALY (Extreme Latency)"
    
    # 2. HEAVY LOAD (SPIKE)
    if c_cpu > global_stats['CPU_p90'] or c_qps > global_stats['QPS_p95']:
        if c_lat > global_stats['Lat_p75']: 
            return "HEAVY LOAD (Spike+Lag)"
        else:
            return "HEAVY TRAFFIC (Processing)"
    
    # 3. CRITICAL FAILURE
    if c_lat > global_stats['Lat_p95'] or c_err > (global_stats['Err_mean'] + 5.0):
        return "CRITICAL FAILURE"

    # 4. ELEVATED
    if c_lat > global_stats['Lat_p75'] or c_cpu > global_stats['CPU_p75']:
        return "ELEVATED LOAD"

    # 5. NORMAL
    return "NORMAL"

def calculate_global_stats(df):
    stats = {
        'Lat_p75': df['LatencyMs'].quantile(0.75),
        'Lat_p95': df['LatencyMs'].quantile(0.95),
        'Lat_p99': df['LatencyMs'].quantile(0.99),
        
        'CPU_p75': df['CpuUsage'].quantile(0.75),
        'CPU_p90': df['CpuUsage'].quantile(0.90),
        
        'QPS_p95': df['LocalQps'].quantile(0.95),
        
        'Err_mean': df['NetworkErrors'].mean()
    }
    return stats

def analyze_and_print_results(df, k, global_stats):
    print(f"\n{'#'*160}")
    print(f" WYNIKI K-MEANS (K={k})")
    print(f"{'#'*160}")

    # Agregacja
    summary = df.groupby('Cluster').agg(
        # Cechy
        Avg_Lat=('LatencyMs', 'mean'),
        Avg_CPU=('CpuUsage', 'mean'),
        Avg_Mem=('MemoryUsageMb', 'mean'),
        Avg_Disk=('DiskQueueLength', 'mean'),
        Avg_NetErr=('NetworkErrors', 'mean'),
        Avg_QPS=('LocalQps', 'mean'),
        Avg_Req=('RequestSizeBytes', 'mean'),
        Avg_Resp=('ResponseSizeBytes', 'mean'),
        
        # Weryfikacja (Ground Truth)
        IsAnom_Pct=('IsAnomaly', lambda x: np.mean(x) * 100),
        Warn_Pct=('Priority', lambda x: (x == 'warn').mean() * 100),
        Err_Pct=('Priority', lambda x: (x == 'err').mean() * 100),
        Crit_Pct=('Priority', lambda x: (x == 'crit').mean() * 100),
        SpikeLat_Pct=('IsSpikeByLatency', lambda x: np.mean(x) * 100),
        SpikeCpu_Pct=('IsSpikeByCpu', lambda x: np.mean(x) * 100),
        
        Top_Event=('EventCode', lambda x: x.mode()[0] if not x.mode().empty else 0),
        Count=('LatencyMs', 'count')
    ).reset_index()

    # Dynamiczna interpretacja
    summary['Interpretation'] = summary.apply(lambda row: interpret_cluster_dynamic(row, global_stats), axis=1)

    # Wyświetlanie
    print(f"{'ID':<3} | {'INTERPRETACJA':<23} | {'Count':<7} || "
          f"{'Lat':<6} | {'CPU':<4} | {'Mem':<5} | {'Disk':<4} | {'NetE':<4} | {'QPS':<5} || "
          f"{'Anom%':<5} | {'Crit%':<5} || "
          f"{'SpikeL%':<7} | {'SpikeC%':<7} | {'Event':<5}")
    print("-" * 140)
    
    for _, row in summary.iterrows():
        print(f"{int(row['Cluster']):<3} | {row['Interpretation']:<23} | {int(row['Count']):<7} || "
              f"{row['Avg_Lat']:<6.0f} | {row['Avg_CPU']:<4.0f} | {row['Avg_Mem']:<5.0f} | {row['Avg_Disk']:<4.1f} | {row['Avg_NetErr']:<4.1f} | {row['Avg_QPS']:<5.0f} || "
              f"{row['IsAnom_Pct']:<5.1f} | {row['Crit_Pct']:<5.1f} || "
              f"{row['SpikeLat_Pct']:<7.1f} | {row['SpikeCpu_Pct']:<7.1f} | {int(row['Top_Event']):<5}")
    
    print("-" * 140)

def run_clustering():
    if not PARQUET_DIR.exists():
        print(f"BŁĄD: Nie znaleziono {PARQUET_DIR}")
        sys.exit(1)
        
    start_time = time.time()
    con = duckdb.connect()
    parquet_path_str = str(PARQUET_DIR).replace('\\', '/')
    
    print(f"Start K-Means (5.9).")
    
    # 1. Pobieranie danych
    feature_cols = [
        'LatencyMs', 'CpuUsage', 'MemoryUsageMb', 'DiskQueueLength', 
        'NetworkErrors', 'LocalQps', 'RequestSizeBytes', 'ResponseSizeBytes'
    ]
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
        print(f"Błąd: {e}")
        return

    # 2. Obliczanie statystyk globalnych (BASELINE)
    global_stats = calculate_global_stats(df)
    
    print(f"Pobrano {len(df)} wierszy.")
    print("Wyznaczone progi dynamiczne (Baseline):")
    print(f"  Lat p95: {global_stats['Lat_p95']:.1f} ms, Lat p99: {global_stats['Lat_p99']:.1f} ms")
    print(f"  CPU p90: {global_stats['CPU_p90']:.1f} %, QPS p95: {global_stats['QPS_p95']:.1f}")

    # 3. Preprocessing
    X = df[feature_cols].copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. K-Means
    k_values = [3, 5, 8]
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        analyze_and_print_results(df, k, global_stats)

    print(f"\nAnaliza zakończona w: {time.time() - start_time:.2f} s")

if __name__ == "__main__":
    run_clustering()