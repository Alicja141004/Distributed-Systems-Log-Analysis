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
SAMPLE_SIZE = 250000 

def get_cluster_label(row, stats):
    """
    Nadaje etykietę klastrowi w oparciu o jego statystyki względem reszty grup.
    stats: słownik ze średnimi i odchyleniami dla całej populacji klastrów
    """
    labels = []
    
    # 1. DETEKCJA AWARII (Bezwzględna)
    # Jeśli >10% anomalii lub błędy krytyczne, to na pewno awaria.
    if row['Anomaly_Rate'] > 10.0 or row['Top_Event'] in [500, 501, 998, 999]:
        return "CRITICAL FAILURE"
    
    # Jeśli dużo błędów sieciowych (> 5%), to problem sieciowy
    if row['Avg_NetErr'] > 5.0:
        return "NETWORK ISSUES"

    # 2. OCENA ZASOBÓW (Relatywna - względem innych klastrów / Obliczamy Z-Score)
    cpu_z = (row['Avg_CPU'] - stats['cpu_mean']) / stats['cpu_std'] if stats['cpu_std'] > 0 else 0
    lat_z = (row['Avg_Latency'] - stats['lat_mean']) / stats['lat_std'] if stats['lat_std'] > 0 else 0
    
    # Progi dynamiczne
    is_high_cpu = cpu_z > 0.8
    is_high_lat = lat_z > 0.8
    
    if is_high_cpu and is_high_lat:
        return "HEAVY LOAD (CPU+LAT)"
    elif is_high_cpu:
        return "HIGH CPU LOAD"
    elif is_high_lat:
        return "HIGH LATENCY (LAG)"
        
    # 3. OCENA NORMALNEGO RUCHU
    if row['Avg_Latency'] < stats['lat_mean']:
        return "NORMAL (Low Traffic)"
    else:
        return "NORMAL (Medium Traffic)"

def analyze_clusters(df, k):
    # 1. Agregacja centroidów
    summary = df.groupby('Cluster').agg(
        Count=('LatencyMs', 'count'),
        Avg_Latency=('LatencyMs', 'mean'),
        Avg_CPU=('CpuUsage', 'mean'),
        Avg_NetErr=('NetworkErrors', 'mean'),
        Anomaly_Rate=('IsAnomaly', lambda x: np.mean(x) * 100),
        Top_Event=('EventCode', lambda x: x.mode()[0] if not x.mode().empty else 0)
    ).reset_index()

    # 2. Obliczamy statystyki globalne dla centroidów (żeby mieć punkt odniesienia)
    # Wykluczamy ewidentne awarie z obliczania "normy", żeby nie zawyżały średniej
    non_failures = summary[
        (summary['Anomaly_Rate'] < 10) & 
        (~summary['Top_Event'].isin([500, 501, 998, 999]))
    ]
    
    if non_failures.empty:
        non_failures = summary

    stats = {
        'cpu_mean': non_failures['Avg_CPU'].mean(),
        'cpu_std': non_failures['Avg_CPU'].std() or 1.0,
        'lat_mean': non_failures['Avg_Latency'].mean(),
        'lat_std': non_failures['Avg_Latency'].std() or 1.0
    }

    # 3. Nadajemy etykiety w pętli
    summary['Label'] = summary.apply(lambda row: get_cluster_label(row, stats), axis=1)
    
    return summary.sort_values(by='Avg_Latency')

def run_clustering():
    if not PARQUET_DIR.exists():
        print(f"BŁĄD: Nie znaleziono {PARQUET_DIR}")
        sys.exit(1)
        
    start_time = time.time()
    con = duckdb.connect()
    parquet_path_str = str(PARQUET_DIR).replace('\\', '/')
    parquet_query = f"read_parquet('{parquet_path_str}/**/*.parquet', hive_partitioning=true)"

    print(f"Start K-Means (5.9). Pobieranie próbki {SAMPLE_SIZE} wierszy...")

    query = f"""
    SELECT 
        LatencyMs, CpuUsage, DiskQueueLength, NetworkErrors, LocalQps,
        EventCode, IsAnomaly
    FROM {parquet_query}
    WHERE LatencyMs IS NOT NULL AND CpuUsage IS NOT NULL AND ScenarioStepIndex IS NOT NULL
    USING SAMPLE {SAMPLE_SIZE}
    """
    
    try:
        df = con.execute(query).df()
    except Exception as e:
        print(f"Błąd: {e}")
        return

    print(f"Pobrano {len(df)} wierszy. Czas: {time.time()-start_time:.2f}s")

    # Standaryzacja
    feature_cols = ['LatencyMs', 'CpuUsage', 'DiskQueueLength', 'NetworkErrors', 'LocalQps']
    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PĘTLA K = [3, 5, 8]
    k_values = [3, 5, 8]
    
    for k in k_values:
        print(f"\n{'='*80}")
        print(f" ANALIZA DLA K = {k}")
        print(f"{'='*80}")
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        summary = analyze_clusters(df, k)
        
        print(f"{'Clust':<5} | {'Count':<8} | {'Lat(ms)':<8} | {'CPU(%)':<6} | {'NetErr':<6} | {'Anom%':<6} | {'Event':<5} | INTERPRETACJA")
        print("-" * 105)
        
        for _, row in summary.iterrows():
            print(f"{int(row['Cluster']):<5} | {int(row['Count']):<8} | {row['Avg_Latency']:<8.1f} | {row['Avg_CPU']:<6.1f} | {row['Avg_NetErr']:<6.2f} | {row['Anomaly_Rate']:<6.1f} | {int(row['Top_Event']):<5} | {row['Label']}")

    print(f"\nCałość wykonana w: {time.time() - start_time:.2f} s")

if __name__ == "__main__":
    run_clustering()