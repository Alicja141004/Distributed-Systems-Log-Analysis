from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"

class EventPredictor:
    # =========================================================================
    # KONFIGURACJA TUNINGU (Stałe konfiguracyjne)
    # =========================================================================
    CONFIG = {
        # --- Ustawienia ogólne ---
        'TRAIN_TEST_SPLIT_RATIO': 0.70,  # 70% trening, 30% test
        
        # --- Progi "Podłogi Bezpieczeństwa" (Safety Floors) ---
        'CPU_FLOOR': 80,             # Minimalne CPU, żeby uznać za Overload
        'DISK_FLOOR': 20,            # Minimalna kolejka dysku
        'MEM_USAGE_FLOOR': 100,      # Minimalne zużycie RAM (MB)
        'LATENCY_FLOOR_DEADLOCK': 1000, # Minimalne opóźnienie dla Deadlocka (ms)
        'NET_ERR_FLOOR': 5,          # Ignoruj mniej niż 5 błędów sieciowych
        
        # --- Wykrywanie Trendów (Delt) ---
        'MEM_LEAK_DIFF_MB': 50.0,    # Wzrost RAM o tyle MB między logami to wyciek
        
        # --- Mnożniki (Statystyka vs Rzeczywistość) ---
        'OVERLOAD_LATENCY_MULT': 1.5, # Overload = High CPU + Latency > Mediana * 1.5
        'DEADLOCK_LATENCY_MULT': 2.0, # Deadlock = Latency > p99 * 2.0
        
        # --- Anomalie Metryczne (Ile razy norma?) ---
        'ANOMALY_NET_MULT': 3.0,     # Anomalia sieci = p95 * 3.0
        'ANOMALY_DISK_MULT': 2.0,    # Anomalia dysku = p95 * 2.0
        
        # --- Regresja (Kategorie obciążenia) ---
        'REG_CRITICAL_NET': 5,       # Błędy sieci > 5 -> Critical
        'REG_CRITICAL_DISK': 80,     # Dysk > 80 -> Critical
        'REG_HEAVY_CPU': 90,         # CPU > 90 -> Heavy Load
        
        # --- Sekwencyjna (Wagi kar) ---
        'SEQ_WEIGHT_ERROR': 3.0,     # Waga za błąd 500 w historii
        'SEQ_WEIGHT_LATENCY': 1.0,   # Waga za wzrost latencji
        'SEQ_THRESHOLD': 0.4,        # Próg alarmu (np. 0.4 = 2 punkty na 5)
        
        # --- Próbkowanie ---
        'LIMIT_BALANCED_SAMPLES': 500, # Ile błędów pobrać do testu
        'LIMIT_NORMAL_SAMPLES': 4500,  # Ile normalnych logów pobrać
    }
    # =========================================================================

    def __init__(self):
        self.con = duckdb.connect()
        self.con.execute("PRAGMA threads=8; PRAGMA memory_limit='16GB'")
        self.stats = {}
        self.split_time = None
        self.cfg = self.CONFIG

    def determine_split(self):
        print("Obliczanie punktu podziału")
        times = self.con.execute(f"""
            SELECT MIN(Timestamp), MAX(Timestamp) 
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
        """).fetchone()
        
        if not times or times[0] is None:
            raise ValueError("Brak danych lub kolumny Timestamp")

        start, end = times
        total_seconds = (end - start).total_seconds()
        self.split_time = start + pd.Timedelta(seconds=total_seconds * self.cfg['TRAIN_TEST_SPLIT_RATIO'])
        print(f"Zakres danych: {start} -> {end}")
        print(f"Punkt podziału (Split Time): {self.split_time}")

    def load_baseline_stats(self):
        if not self.split_time: self.determine_split()
        
        print("Ładowanie statystyk bazowych")
        row = self.con.execute(f"""
            SELECT 
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY LatencyMs),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY LatencyMs),
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY LatencyMs),
                
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY CpuUsage),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY DiskQueueLength),
                
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY MemoryUsageMb),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY NetworkErrors),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY Retries),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY LocalQps)
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
            WHERE LatencyMs IS NOT NULL 
              AND Timestamp < ?
        """, [self.split_time]).fetchone()

        keys = [
            'lat_median', 'lat_p95', 'lat_p99', 
            'cpu_p95', 'disk_p95', 
            'mem_p95', 'net_p95', 'retry_p95', 'qps_p95'
        ]
        defaults = [50, 300, 1000, 90, 50, 2048, 5, 0, 100]
        self.stats = {k: (v if v is not None else d) for k, v, d in zip(keys, row, defaults)}

        print("Budowanie profilu znanych kodów zdarzeń")
        code_counts = self.con.execute(f"""
            SELECT EventCode, COUNT(*) as Cnt
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
            WHERE Timestamp < ?
            GROUP BY EventCode
        """, [self.split_time]).df()
        
        self.stats['known_codes'] = set(code_counts['EventCode'].tolist())
        total_train_events = code_counts['Cnt'].sum()
        rare_df = code_counts[code_counts['Cnt'] / total_train_events < 0.001]
        self.stats['rare_codes'] = set(rare_df['EventCode'].tolist())
        
        self._print_thresholds()

        return self.stats

    def _print_thresholds(self):
        """Wypisuje obliczone progi i warunki alarmowe."""
        s = self.stats
        c = self.cfg
        
        print("\n" + "="*60)
        print("PROGI DETEKCJI AWARII (NAUCZONE + SKONFIGUROWANE)")
        print("="*60)
        
        print(f"1. OVERLOAD (Przeciążenie):")
        print(f"   - CPU > {s['cpu_p95']:.1f}% (p95) ORAZ > {c['CPU_FLOOR']}% (Floor)")
        print(f"   - LUB Dysk Queue > {s['disk_p95']:.1f} (p95) ORAZ > {c['DISK_FLOOR']} (Floor)")
        print(f"   - WARUNEK KONIECZNY: Opóźnienie > {s['lat_median'] * c['OVERLOAD_LATENCY_MULT']:.1f}ms (Mediana * {c['OVERLOAD_LATENCY_MULT']})")
        
        print(f"\n2. MEMORY LEAK (Wyciek Pamięci):")
        print(f"   - Zużycie RAM > {s['mem_p95']:.1f} MB (p95) ORAZ > {c['MEM_USAGE_FLOOR']} MB (Floor)")
        print(f"   - Wzrost (Delta) > {c['MEM_LEAK_DIFF_MB']} MB pomiędzy logami")
        
        print(f"\n3. NETWORK ISSUES (Sieć):")
        print(f"   - Błędy sieci > {s['net_p95']:.1f} (p95) ORAZ > {c['NET_ERR_FLOOR']} (Floor)")
        print(f"   - LUB Retries > 0 ORAZ Błędy sieci > 2")
        
        print(f"\n4. DEADLOCK (Zakleszczenie):")
        print(f"   - Opóźnienie > {s['lat_p99'] * c['DEADLOCK_LATENCY_MULT']:.1f}ms (p99 * {c['DEADLOCK_LATENCY_MULT']})")
        print(f"   - ORAZ Opóźnienie > {c['LATENCY_FLOOR_DEADLOCK']}ms (Floor)")
        print(f"   - ORAZ CPU < 5% i Dysk < 2")
        
        print(f"\n5. ANOMALIE METRYCZNE:")
        print(f"   - Sieć > {s['net_p95'] * c['ANOMALY_NET_MULT']:.1f} (p95 * {c['ANOMALY_NET_MULT']})")
        print(f"   - Dysk > {s['disk_p95'] * c['ANOMALY_DISK_MULT']:.1f} (p95 * {c['ANOMALY_DISK_MULT']})")
        
        print(f"\n6. ANOMALIE LOGICZNE:")
        print(f"   - Znane kody: {len(s['known_codes'])}")
        print(f"   - Rzadkie kody (<0.1%): {len(s['rare_codes'])}")
        print("="*60 + "\n")

    def predict_failure_and_anomaly(self):
        print("Test Klasyfikacji")
        
        query_enhanced = f"""
        WITH Trends AS (
            SELECT *,
                MemoryUsageMb - LAG(MemoryUsageMb, 1, MemoryUsageMb) OVER (
                    PARTITION BY SystemRole, ClusterNode 
                    ORDER BY Timestamp
                ) as Mem_Diff
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true) 
            WHERE Timestamp >= '{self.split_time}'
        )
        SELECT * FROM (
            (SELECT * FROM Trends WHERE EventCode IN (500, 501, 998, 999) LIMIT {self.cfg['LIMIT_BALANCED_SAMPLES']})
            UNION ALL
            (SELECT * FROM Trends WHERE EventCode NOT IN (500, 501, 998, 999) LIMIT {self.cfg['LIMIT_NORMAL_SAMPLES']})
        )
        """
        df = self.con.execute(query_enhanced).df()
        
        if df.empty:
            print("Brak danych w zbiorze testowym!")
            return pd.DataFrame()

        # --- SCENARIUSZE AWARII ---
        
        # 1. Overload
        is_overload = (
            (
                (df['CpuUsage'] > self.stats['cpu_p95']) & (df['CpuUsage'] > self.cfg['CPU_FLOOR'])
            ) | (
                (df['DiskQueueLength'] > self.stats['disk_p95']) & (df['DiskQueueLength'] > self.cfg['DISK_FLOOR'])
            )
        ) & (df['LatencyMs'] > self.stats['lat_median'] * self.cfg['OVERLOAD_LATENCY_MULT'])

        # 2. Memory Leak
        mem_diff = df['Mem_Diff'].fillna(0)
        is_mem_leak = (
            (df['MemoryUsageMb'] > self.stats['mem_p95']) & 
            (mem_diff > self.cfg['MEM_LEAK_DIFF_MB']) & 
            (df['MemoryUsageMb'] > self.cfg['MEM_USAGE_FLOOR'])
        )

        # 3. Network Issues
        is_network = (
            (df['NetworkErrors'] > self.stats['net_p95']) & 
            (df['NetworkErrors'] > self.cfg['NET_ERR_FLOOR'])
        ) | (
            (df['Retries'] > 0) & 
            (df['NetworkErrors'] > 2)
        )

        # 4. Deadlock
        lat = df['LatencyMs'].fillna(0)
        is_deadlock = (
            (lat > self.stats['lat_p99'] * self.cfg['DEADLOCK_LATENCY_MULT']) &
            (lat > self.cfg['LATENCY_FLOOR_DEADLOCK']) &
            (df['CpuUsage'] < 5) &                
            (df['DiskQueueLength'] < 2)
        )

        # 5. Hard Failure
        is_hard_fail = df['EventCode'].isin([500, 501, 998, 999])

        predicted_failure = is_overload | is_mem_leak | is_network | is_deadlock | is_hard_fail

        # --- ANOMALIE ---
        is_metric_anomaly = (
            ((df['NetworkErrors'] > self.stats['net_p95'] * self.cfg['ANOMALY_NET_MULT']) & (df['NetworkErrors'] > 10)) |
            ((df['DiskQueueLength'] > self.stats['disk_p95'] * self.cfg['ANOMALY_DISK_MULT']) & (df['DiskQueueLength'] > 50))
        )
        
        known_set = self.stats['known_codes']
        rare_set = self.stats['rare_codes']
        is_logical_anomaly = df['EventCode'].apply(lambda x: x not in known_set or x in rare_set)
        
        predicted_anomaly = is_metric_anomaly | is_logical_anomaly

        return pd.DataFrame({
            'Predicted_Failure': predicted_failure,
            'Actual_Failure': df['EventCode'].isin([500, 501, 998, 999]),
            'Predicted_Anomaly': predicted_anomaly,
            'Actual_Anomaly': (df['IsAnomaly'] == 1) | is_logical_anomaly
        })

    def predict_latency_simple(self):
        print("Test Regresji")
        df = self.con.execute(f"""
            SELECT 
                LatencyMs, CpuUsage, DiskQueueLength, NetworkErrors, Retries,
                CASE 
                    WHEN NetworkErrors > {self.cfg['REG_CRITICAL_NET']} OR Retries > 0 OR DiskQueueLength > {self.cfg['REG_CRITICAL_DISK']} THEN 'critical'
                    WHEN CpuUsage > {self.cfg['REG_HEAVY_CPU']} OR DiskQueueLength > 50 THEN 'heavy_load'
                    WHEN CpuUsage > 70 THEN 'medium_load'
                    ELSE 'normal'
                END as predicted_category
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
            WHERE LatencyMs IS NOT NULL 
              AND Timestamp >= '{self.split_time}'
            USING SAMPLE 20% LIMIT 3000
        """).df()

        if df.empty: return pd.DataFrame()

        mapping = {
            'normal':      self.stats['lat_median'],       
            'medium_load': self.stats['lat_p95'] * 0.7,
            'heavy_load':  self.stats['lat_p95'] * 1.2,          
            'critical':    self.stats['lat_p99']           
        }
        
        df['Predicted'] = df['predicted_category'].map(mapping)
        df['Error'] = (df['Predicted'] - df['LatencyMs']).abs()
        
        return df[['Predicted', 'LatencyMs', 'Error']]

    def predict_sequential_simple(self):
        print("Analiza sekwencyjna (Szukanie transakcji problematycznych)...")
        
        tx_df = self.con.execute(f"""
            SELECT TransactionId 
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
            WHERE Timestamp >= '{self.split_time}'
            GROUP BY TransactionId 
            HAVING COUNT(*) >= 3 
               AND (MAX(EventCode) >= 500 OR RANDOM() < 0.05)
            LIMIT 200
        """).df()
        
        if tx_df.empty: return pd.DataFrame()

        ids_fmt = "'" + "','".join(tx_df['TransactionId'].tolist()) + "'"
        events_df = self.con.execute(f"""
            SELECT TransactionId, EventCode, LatencyMs, CpuUsage, DiskQueueLength, 
                   NetworkErrors, Retries, IsAnomaly, Timestamp 
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true) 
            WHERE TransactionId IN ({ids_fmt}) 
            ORDER BY TransactionId, Timestamp
        """).df()

        predictions = []
        for _, grp in events_df.groupby('TransactionId'):
            grp = grp.reset_index(drop=True)
            for i in range(2, len(grp)):
                hist = grp.iloc[:i]
                warnings = 0
                
                # Zastosowanie wag z konfiguracji
                if (hist['EventCode'] >= 500).any(): 
                    warnings += self.cfg['SEQ_WEIGHT_ERROR']
                
                if len(hist) >= 2:
                    if hist['LatencyMs'].iloc[-1] > hist['LatencyMs'].mean() * 1.5:
                        warnings += self.cfg['SEQ_WEIGHT_LATENCY']
                
                if (hist['Retries'] > 0).any(): 
                    warnings += 1.0

                is_prob = (grp.iloc[i]['EventCode'] >= 500) or (grp.iloc[i]['IsAnomaly'] == 1)
                
                score = min(warnings / 5.0, 1.0)
                
                predictions.append({
                    'Score': score, 
                    'Predicted_Problem': score >= self.cfg['SEQ_THRESHOLD'],
                    'Actual_Problem': is_prob
                })
        
        return pd.DataFrame(predictions)

    def _print_metrics(self, df, pred_col, actual_col, name):
        if df.empty: 
            print(f"{name}: Brak danych do oceny.")
            return
        tp = (df[pred_col] & df[actual_col]).sum()
        fp = (df[pred_col] & ~df[actual_col]).sum()
        fn = (~df[pred_col] & df[actual_col]).sum()
        tn = (~df[pred_col] & ~df[actual_col]).sum()
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0
        acc = (tp + tn) / len(df)
        print(f"{name} F1: {f1:.3f} | Acc: {acc:.2f} (TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn})")

    def run(self):
        self.load_baseline_stats()
        f = self.predict_failure_and_anomaly()
        l = self.predict_latency_simple()
        s = self.predict_sequential_simple()
        
        print("\n--- WYNIKI ---")
        self._print_metrics(f, 'Predicted_Failure', 'Actual_Failure', 'Awaria')
        self._print_metrics(f, 'Predicted_Anomaly', 'Actual_Anomaly', 'Anomalia')
        
        if not l.empty:
            print("\n--- REGRESJA OPÓŹNIEŃ ---")
            print(f"Średni Błąd Bezwzględny (MAE): {l['Error'].mean():.1f}ms")
            mse = (l['Error'] ** 2).mean()
            print(f"Błąd Średniokwadratowy (MSE): {mse:.1f}")

        print("")
        self._print_metrics(s, 'Predicted_Problem', 'Actual_Problem', 'Sekwencyjna')

def main():
    if not PARQUET_DIR.exists(): return print("Brak danych.")
    try: EventPredictor().run()
    except Exception as e: 
        import traceback
        traceback.print_exc()
        print(f"Błąd: {e}")

if __name__ == "__main__": main()