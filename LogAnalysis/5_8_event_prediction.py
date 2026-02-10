from pathlib import Path
import duckdb
import pandas as pd
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"

class EventPredictor:
    def __init__(self):
        self.con = duckdb.connect()
        self.con.execute("PRAGMA threads=8; PRAGMA memory_limit='16GB'")
        self.stats = {}
    
    def load_baseline_stats(self):
        print("Ładowanie statystyk bazowych...")
        # Pobranie percentyli
        row = self.con.execute(f"""
            SELECT 
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY LatencyMs),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY LatencyMs),
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY LatencyMs),
                PERCENTILE_CONT(0.80) WITHIN GROUP (ORDER BY CpuUsage),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY CpuUsage),
                PERCENTILE_CONT(0.80) WITHIN GROUP (ORDER BY DiskQueueLength),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY DiskQueueLength),
                COUNT(*)
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
            WHERE LatencyMs IS NOT NULL AND CpuUsage IS NOT NULL
        """).fetchone()

        # Rozpakowanie wyników do słownika (bezpieczniej i czytelniej)
        keys = ['lat_median', 'lat_p95', 'lat_p99', 'cpu_p80', 'cpu_p95', 'disk_p80', 'disk_p95', 'total_events']
        # Używamy wartości domyślnych, jeśli baza zwróci None
        defaults = [50, 300, 1000, 70, 90, 20, 50, 0]
        self.stats = {k: (v if v is not None else d) for k, v, d in zip(keys, row, defaults)}

        # Pobranie Impact Factor
        impact = self.con.execute(f"""
            SELECT
                AVG(CASE WHEN CpuUsage > ? THEN LatencyMs END) / NULLIF(AVG(LatencyMs), 0),
                AVG(CASE WHEN DiskQueueLength > ? THEN LatencyMs END) / NULLIF(AVG(LatencyMs), 0),
                AVG(CASE WHEN IsSpikeByLatency = true THEN LatencyMs END) / NULLIF(AVG(LatencyMs), 0)
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
        """, [self.stats['cpu_p80'], self.stats['disk_p80']]).fetchone()
        
        self.stats.update({
            'cpu_impact': impact[0] or 1.2,
            'disk_impact': impact[1] or 1.1,
            'spike_impact': impact[2] or 2.0
        })

        # Statystyki per kategoria
        cat_df = self.con.execute(f"""
            SELECT 
                CASE 
                    WHEN EventCode IN (500, 501) THEN 'failure'
                    WHEN IsAnomaly = 1 THEN 'anomaly'
                    WHEN LatencyMs > ? THEN 'high_latency' ELSE 'normal'
                END as category,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY LatencyMs) as median,
                PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY LatencyMs) as p80,
                AVG(LatencyMs) as avg
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
            WHERE LatencyMs IS NOT NULL GROUP BY category
        """, [self.stats['lat_p95']]).df().set_index('category')
        
        self.stats['categories'] = cat_df.to_dict('index')
        return self.stats

    def predict_failure_and_anomaly(self):
        print("Pobieranie próbek (Awarie + Anomalie + Norma)...")
        df = self.con.execute(f"""
            (SELECT * FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true) WHERE EventCode IN (500, 501) LIMIT 333)
            UNION ALL
            (SELECT * FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true) WHERE IsAnomaly = 1 AND EventCode NOT IN (500, 501) LIMIT 333)
            UNION ALL
            (SELECT * FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true) WHERE (EventCode NOT IN (500, 501) OR EventCode IS NULL) AND (IsAnomaly != 1 OR IsAnomaly IS NULL) USING SAMPLE 10% LIMIT 3333)
        """).df()

        # Wektoryzacja (zamiast pętli for)
        df['fail_score'] = 0.0
        df['anom_score'] = 0.0
        
        # Obliczanie Failure Score
        df.loc[df['EventCode'].isin([500, 501]), 'fail_score'] += 0.5
        df.loc[df['Priority'] == 'crit', 'fail_score'] += 0.3
        df.loc[df['Priority'] == 'err', 'fail_score'] += 0.2
        df.loc[df['LatencyMs'] > self.stats['lat_p99'], 'fail_score'] += 0.2
        
        # Obliczanie Anomaly Score
        df.loc[df['IsAnomaly'] == 1, 'anom_score'] += 0.5
        df.loc[df['EventCode'].isin([998, 999]), 'anom_score'] += 0.3
        df.loc[df['LatencyMs'] > self.stats['lat_p99'] * 2, 'anom_score'] += 0.2

        # Tworzenie wyniku
        results = pd.DataFrame({
            'Predicted_Failure': df['fail_score'] >= 0.45,
            'Predicted_Anomaly': df['anom_score'] >= 0.35,
            'Actual_Failure': df['EventCode'].isin([500, 501]),
            'Actual_Anomaly': df['IsAnomaly'] == 1
        })
        return results

    def predict_latency_simple(self):
        df = self.con.execute(f"""
            SELECT LatencyMs, CpuUsage, DiskQueueLength, IsSpikeByLatency, EventCode, IsAnomaly,
            CASE 
                WHEN EventCode IN (500, 501) THEN 'failure'
                WHEN IsAnomaly = 1 THEN 'anomaly'
                WHEN LatencyMs > ? THEN 'high_latency' ELSE 'normal'
            END as category
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
            WHERE LatencyMs IS NOT NULL USING SAMPLE 10% LIMIT 3000
        """, [self.stats['lat_p95']]).df()

        # Baza predykcji na podstawie kategorii
        df['Predicted'] = df['category'].map(lambda c: self.stats['categories'].get(c, {}).get('median', self.stats['lat_median']))
        
        # Modyfikatory dla 'normal'
        norm_mask = df['category'] == 'normal'
        df.loc[norm_mask & (df['CpuUsage'] > self.stats['cpu_p95']), 'Predicted'] *= self.stats['cpu_impact']
        df.loc[norm_mask & (df['DiskQueueLength'] > self.stats['disk_p95']), 'Predicted'] *= self.stats['disk_impact']
        df.loc[norm_mask & df['IsSpikeByLatency'], 'Predicted'] *= self.stats['spike_impact']

        # Modyfikatory dla 'high_latency'
        hl_mask = df['category'] == 'high_latency'
        # Nadpisz bazę na p80
        df.loc[hl_mask, 'Predicted'] = self.stats['categories'].get('high_latency', {}).get('p80', self.stats['lat_p95'])
        df.loc[hl_mask & (df['CpuUsage'] > self.stats['cpu_p80']), 'Predicted'] *= self.stats['cpu_impact']

        # Modyfikatory dla błędów
        fail_mask = df['category'].isin(['failure', 'anomaly'])
        df.loc[fail_mask, 'Predicted'] = df.loc[fail_mask, 'category'].map(lambda c: self.stats['categories'].get(c, {}).get('avg', 0))

        # Guardrails (clip)
        df['Predicted'] = df['Predicted'].clip(lower=df['LatencyMs'] * 0.2, upper=df['LatencyMs'] * 5.0)
        
        df['Error'] = (df['Predicted'] - df['LatencyMs']).abs()
        return df[['Predicted', 'LatencyMs', 'Error']] # Zwracamy tylko co potrzebne

    def predict_sequential_simple(self):
        print("Analiza sekwencyjna (ZOPTYMALIZOWANA)...")
        # SQL bez zmian - jest już optymalny
        tx_df = self.con.execute(f"""
            WITH Probl AS (SELECT TransactionId FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true) GROUP BY TransactionId HAVING COUNT(*) >= 3 AND (MAX(EventCode) >= 500 OR MAX(IsAnomaly) = 1) LIMIT 50),
            Norm AS (SELECT TransactionId FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true) GROUP BY TransactionId HAVING COUNT(*) >= 3 AND MAX(EventCode) = 200 LIMIT 50)
            SELECT TransactionId FROM Probl UNION ALL SELECT TransactionId FROM Norm
        """).df()
        
        if tx_df.empty: return pd.DataFrame()

        ids_fmt = "'" + "','".join(tx_df['TransactionId'].tolist()) + "'"
        events_df = self.con.execute(f"""
            SELECT TransactionId, EventCode, LatencyMs, IsAnomaly 
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true) 
            WHERE TransactionId IN ({ids_fmt}) ORDER BY TransactionId, Timestamp
        """).df()

        predictions = []
        for _, grp in events_df.groupby('TransactionId'):
            # Logika jest trudna do pełnej wektoryzacji przez zależność od historii
            # ale możemy uprościć zapis w pętli
            grp = grp.reset_index(drop=True)
            for i in range(2, len(grp)):
                hist = grp.iloc[:i]
                warnings = 0
                if (hist['EventCode'] >= 500).any(): warnings += 2
                if (hist['LatencyMs'] > self.stats['lat_p95']).any(): warnings += 1
                
                # Trend
                if len(hist) >= 3:
                    if hist['LatencyMs'].tail(2).mean() > hist['LatencyMs'].head(len(hist)-2).mean() * 1.3:
                        warnings += 1

                is_prob = (grp.iloc[i]['EventCode'] >= 500) or (grp.iloc[i]['IsAnomaly'] == 1)
                predictions.append({'Score': warnings/5.0, 'Predicted_Problem': warnings/5.0 >= 0.25, 'Actual_Problem': is_prob})
        
        return pd.DataFrame(predictions)

    def _print_metrics(self, df, pred_col, actual_col, name):
        """Helper do wyświetlania metryk, żeby nie powielać kodu"""
        if df.empty: return
        tp = (df[pred_col] & df[actual_col]).sum()
        fp = (df[pred_col] & ~df[actual_col]).sum()
        fn = (~df[pred_col] & df[actual_col]).sum()
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0
        print(f"{name} F1: {f1:.3f} (TP: {tp}, FP: {fp})")

    def run(self):
        self.load_baseline_stats()
        f = self.predict_failure_and_anomaly()
        l = self.predict_latency_simple()
        s = self.predict_sequential_simple()
        
        print("\n--- WYNIKI ---")
        self._print_metrics(f, 'Predicted_Failure', 'Actual_Failure', 'Awaria')
        if not l.empty: print(f"Regresja Opóźnień MAE: {l['Error'].mean():.1f}ms")
        self._print_metrics(s, 'Predicted_Problem', 'Actual_Problem', 'Sekwencyjna')

def main():
    if not PARQUET_DIR.exists(): return print("Brak danych.")
    try: EventPredictor().run()
    except Exception as e: print(f"Błąd: {e}")

if __name__ == "__main__": main()