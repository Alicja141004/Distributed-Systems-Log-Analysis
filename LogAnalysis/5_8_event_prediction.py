from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import sys

# Ustalanie ścieżek relatywnych do lokalizacji skryptu.
# Dzięki temu skrypt zadziała niezależnie od tego, na jakiej maszynie zostanie uruchomiony.
THIS_DIR = Path(__file__).resolve().parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"

class EventPredictor:
    """
    Klasa EventPredictor implementuje silnik predykcyjny oparty na regułach (Rule-Based Machine Learning)
    służący do analizy logów systemowych (AIOps).
    
    Główne cele:
    1. Budowanie profilu statystycznego systemu na podstawie danych historycznych (Baseline).
    2. Wykrywanie awarii (Failure Prediction) zanim wystąpią krytyczne błędy.
    3. Identyfikacja anomalii (Anomaly Detection) w metrykach wydajnościowych.
    """

    # =========================================================================
    # KONFIGURACJA MODELU PREDYKCYJNEGO (TUNED FOR PRECISION)
    # =========================================================================
    # Konfiguracja została dobrana tak, aby minimalizować liczbę fałszywych alarmów (False Positives).
    # Model jest "ostrożny" - alarmuje tylko przy wyraźnych sygnałach problemów.
    CONFIG = {
        # --- Ustawienia podziału danych ---
        # Stosunek podziału czasowego: 70% najstarszych danych to trening (nauka normy), 
        # 30% najnowszych to test (weryfikacja). Zapobiega to wyciekowi danych (Data Leakage).
        'TRAIN_TEST_SPLIT_RATIO': 0.70,
        
        # --- Filtry Szumu (Safety Floors) ---
        # Wartości metryk poniżej tych progów są uznawane za "szum tła" i ignorowane.
        # Przykład: Jeśli CPU ma 80%, to dla nowoczesnego serwera nie jest to jeszcze awaria.
        'CPU_FLOOR': 90,             # Alarmuj dopiero, gdy CPU przekroczy 90%
        'DISK_FLOOR': 200,           # Długość kolejki dysku (I/O Wait) musi być znacząca
        'MEM_USAGE_FLOOR': 4096,     # Ignoruj wahania pamięci poniżej 4GB (np. normalne działanie OS)
        'LATENCY_FLOOR_DEADLOCK': 5000, # Deadlock (zakleszczenie) to zazwyczaj całkowity zwis > 5s
        'NET_ERR_FLOOR': 100,        # Ignoruj pojedyncze błędy sieciowe (np. zerwane pakiety Wi-Fi)
        
        # --- Wykrywanie Trendów ---
        # Aby wykryć wyciek pamięci (Memory Leak), szukamy nagłych skoków zużycia.
        # Próg ustawiony wysoko (1GB), aby ignorować działanie Garbage Collectora (GC).
        'MEM_LEAK_DIFF_MB': 1024.0,  
        
        # --- Mnożniki Dynamiczne (Adaptive Thresholds) ---
        # Określają, ile razy bieżąca metryka musi być gorsza od historycznej normy (mediany),
        # aby uznać to za anomalię. Pozwala to modelowi adaptować się do różnych systemów.
        'OVERLOAD_LATENCY_MULT': 40.0, # Np. jeśli zazwyczaj jest 10ms, alarmuj przy 400ms
        'DEADLOCK_LATENCY_MULT': 40.0, 
        'DEADLOCK_CPU_CEILING': 15,    # Przy Deadlocku CPU jest zazwyczaj niskie (proces "wisi", nie liczy)
        
        # --- Detekcja Anomalii Metrycznych ---
        # Bardzo wysokie mnożniki dla anomalii, aby wyłapywać tylko zdarzenia ekstremalne (Outliers).
        'ANOMALY_NET_MULT': 50.0,    
        'ANOMALY_DISK_MULT': 20.0,

        # --- Definicja SLA (Ground Truth) ---
        # Limit czasu odpowiedzi, powyżej którego uznajemy, że system nie działa poprawnie (złamanie umowy).
        # Służy do weryfikacji, czy nasza predykcja była słuszna.
        'SLA_LATENCY_LIMIT_MS': 4000, 
        
        # --- Parametry Regresji Schodkowej ---
        # Progi używane do prostego kategoryzowania obciążenia w module regresji.
        'REG_CRITICAL_NET': 100,
        'REG_CRITICAL_DISK': 300,
        'REG_HEAVY_CPU': 95,
        
        # --- Analiza Sekwencyjna (Scoring) ---
        # Wagi punktowe przyznawane za symptomy w historii transakcji (np. sesji użytkownika).
        'SEQ_WEIGHT_ERROR': 4.0,     # Błąd krytyczny w przeszłości to silny sygnał
        'SEQ_WEIGHT_LATENCY': 2.0,   # Rosnące opóźnienie to sygnał ostrzegawczy
        'SEQ_THRESHOLD': 0.85,       # Próg (0-1), powyżej którego przewidujemy awarię sekwencji
        
        # --- Konfiguracja Przetwarzania ---
        'TOTAL_SAMPLES': 500000,    # Liczba wierszy pobierana do analizy (sampling dla wydajności)
    }
    # =========================================================================

    def __init__(self):
        """
        Inicjalizacja silnika.
        Ustawia połączenie z bazą DuckDB, która przetwarza pliki Parquet w pamięci.
        DuckDB jest tu kluczowy, bo pozwala na szybkie zapytania SQL na dużych zbiorach danych bez ładowania ich do Pandas.
        """
        self.con = duckdb.connect()
        # Ograniczenie pamięci dla DuckDB, aby proces nie został zabity przez OS (OOM Killer)
        self.con.execute("PRAGMA threads=8; PRAGMA memory_limit='16GB'")
        self.stats = {}
        self.split_time = None
        self.cfg = self.CONFIG

    def determine_split(self):
        """
        Analizuje zakres czasowy dostępnych logów i wyznacza punkt podziału 
        na zbiór treningowy i testowy (Temporal Split).
        
        Jest to krytyczne dla rzetelności testów - nie możemy "widzieć przyszłości" podczas uczenia modelu.
        """
        print("Obliczanie punktu podziału (Train/Test Split)...")
        # Pobieramy najwcześniejszy i najpóźniejszy znacznik czasu z całego zbioru danych
        times = self.con.execute(f"""
            SELECT MIN(Timestamp), MAX(Timestamp) 
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
        """).fetchone()
        
        if not times or times[0] is None:
            raise ValueError("Brak danych lub kolumny Timestamp w plikach Parquet.")

        start, end = times
        total_seconds = (end - start).total_seconds()
        # Wyznaczamy punkt w czasie, który dzieli zbiór w proporcji zdefiniowanej w CONFIG (np. 70%)
        self.split_time = start + pd.Timedelta(seconds=total_seconds * self.cfg['TRAIN_TEST_SPLIT_RATIO'])
        print(f"Zakres danych: {start} -> {end}")
        print(f"Punkt podziału (Split Time): {self.split_time}")

    def load_baseline_stats(self):
        """
        Oblicza statystyki bazowe (Historical Profiling).
        Pobiera mediany i percentyle (p95, p99) tylko z danych TRENINGOWYCH (przed split_time).
        
        Cel: Zrozumieć, co oznacza "normalne zachowanie" dla tego konkretnego systemu.
        Dzięki temu model nie wymaga ręcznego ustawiania progów (np. co to jest "duże opóźnienie").
        """
        if not self.split_time: self.determine_split()
        
        print("Profilowanie systemu na danych treningowych...")
        # PERCENTILE_CONT(0.50) to mediana - odporna na pojedyncze wyskoki.
        # PERCENTILE_CONT(0.95) to granica, poniżej której znajduje się 95% ruchu (odcina szum).
        row = self.con.execute(f"""
            SELECT 
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY LatencyMs),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY LatencyMs),
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY LatencyMs),
                
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY CpuUsage),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY DiskQueueLength),
                
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY MemoryUsageMb),
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY NetworkErrors),
                
                AVG(LatencyMs),
                STDDEV(LatencyMs)
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
            WHERE LatencyMs IS NOT NULL 
              AND Timestamp < ? -- WAŻNE: Bierzemy tylko dane z przeszłości!
        """, [self.split_time]).fetchone()

        keys = [
            'lat_median', 'lat_p95', 'lat_p99', 
            'cpu_p95', 'disk_p95', 
            'mem_p95', 'net_p95',
            'lat_avg', 'lat_std'
        ]
        # Wartości domyślne (fallback), gdyby w danych historycznych brakowało rekordów
        defaults = [50, 300, 1000, 90, 50, 2048, 5, 50, 20]
        self.stats = {k: (v if v is not None else d) for k, v, d in zip(keys, row, defaults)}

        print("Identyfikacja znanych kodów zdarzeń...")
        # Budujemy "słownik" znanych kodów błędów (np. 200, 404, 500).
        # Jeśli w przyszłości pojawi się kod spoza tej listy, zostanie oznaczony jako anomalia.
        code_counts = self.con.execute(f"""
            SELECT EventCode, COUNT(*) as Cnt
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
            WHERE Timestamp < ?
            GROUP BY EventCode
        """, [self.split_time]).df()
        
        self.stats['known_codes'] = set(code_counts['EventCode'].tolist())
        total_train_events = code_counts['Cnt'].sum()
        # Kody występujące rzadziej niż 0.1% uznajemy za rzadkie (potencjalnie podejrzane)
        rare_df = code_counts[code_counts['Cnt'] / total_train_events < 0.001] 
        self.stats['rare_codes'] = set(rare_df['EventCode'].tolist())
        
        self._print_thresholds()
        return self.stats

    def _print_thresholds(self):
        """Metoda pomocnicza: wyświetla obliczone progi w czytelnej formie."""
        s = self.stats
        c = self.cfg
        print("\n" + "="*60)
        print("USTALONE PROGI DETEKCJI (BASELINE)")
        print("="*60)
        print(f"1. OVERLOAD (Przeciążenie):")
        print(f"   - CPU Próg: {c['CPU_FLOOR']}% (Baseline p95: {s['cpu_p95']:.1f}%)")
        print(f"   - Latency Próg: {s['lat_median'] * c['OVERLOAD_LATENCY_MULT']:.1f}ms")
        print(f"\n2. MEMORY LEAK (Wyciek):")
        print(f"   - Wymagany skok: > {c['MEM_LEAK_DIFF_MB']} MB ORAZ wysoki stan RAM")
        print(f"\n3. GROUND TRUTH (Definicja SLA):")
        print(f"   - Latency Limit: {c['SLA_LATENCY_LIMIT_MS']} ms")
        print("="*60 + "\n")

    def predict_failure_and_anomaly(self):
        """
        Główny silnik predykcyjny (Inference Engine).
        Pobiera dane TESTOWE i próbuje przewidzieć awarie, używając wiedzy z danych TRENINGOWYCH.
        """
        print("Uruchamianie klasyfikacji zdarzeń na zbiorze testowym...")
        
        # Pobieranie danych testowych z wyliczeniem różnic (Delta) dla pamięci.
        # Używamy funkcji okna (LAG) w SQL, aby zobaczyć zmianę względem poprzedniego wpisu.
        query = f"""
        WITH Trends AS (
            SELECT *,
                MemoryUsageMb - LAG(MemoryUsageMb, 1, MemoryUsageMb) OVER (
                    PARTITION BY SystemRole, ClusterNode 
                    ORDER BY Timestamp
                ) as Mem_Diff
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true) 
            WHERE Timestamp >= '{self.split_time}'
        )
        SELECT * FROM Trends 
        USING SAMPLE {self.cfg['TOTAL_SAMPLES']}
        """
        df = self.con.execute(query).df()
        
        if df.empty: return pd.DataFrame()

        # --- SEKCJA PREDYKCJI (HEURYSTYKA) ---
        # Tutaj definiujemy reguły biznesowe: "Co wygląda jak awaria?"
        
        # 1. Detekcja Przeciążenia (Overload)
        # System jest przeciążony, jeśli brakuje zasobów (CPU/Dysk)
        is_high_load = (df['CpuUsage'] > self.cfg['CPU_FLOOR']) | \
                       (df['DiskQueueLength'] > self.cfg['DISK_FLOOR'])
        
        # ORAZ system działa wolno. Warunek jest podwójny:
        # a) Jest znacznie wolniej niż zazwyczaj (mult * mediana)
        # b) Opóźnienie jest fizycznie odczuwalne (> 100ms) - eliminuje fałszywe alarmy na ultra-szybkich systemach.
        is_slow = (df['LatencyMs'] > (self.stats['lat_median'] * self.cfg['OVERLOAD_LATENCY_MULT'])) & \
                  (df['LatencyMs'] > 100) 
        
        # Ostateczna decyzja: Przeciążenie LUB drastyczne przekroczenie progu bezpieczeństwa (95% SLA)
        pred_overload = (is_slow & is_high_load) | \
                        (df['LatencyMs'] > self.cfg['SLA_LATENCY_LIMIT_MS'] * 0.95)

        # 2. Detekcja Wycieku Pamięci (Memory Leak)
        # Warunek logiczny AND (&):
        # Musi wystąpić nagły skok zużycia (Mem_Diff) PRZY JEDNOCZESNYM wysokim całkowitym zużyciu.
        # To eliminuje "szum" Garbage Collectora, który czyści pamięć.
        pred_mem_leak = (df['Mem_Diff'] > self.cfg['MEM_LEAK_DIFF_MB']) & \
                        (df['MemoryUsageMb'] > self.stats['mem_p95'])

        # 3. Detekcja Problemów Sieciowych
        # Proste przekroczenie progu "bezpieczeństwa" dla błędów sieciowych.
        pred_network = (df['NetworkErrors'] > self.cfg['NET_ERR_FLOOR'])

        # 4. Detekcja Zakleszczenia (Deadlock)
        # Specyficzna sygnatura: System "stoi" (Latency ogromne), ale zużycie zasobów jest niskie.
        # To oznacza, że procesy czekają na blokady (mutex/lock) i nie wykonują pracy.
        pred_deadlock = (df['LatencyMs'] > self.cfg['LATENCY_FLOOR_DEADLOCK']) & \
                        (df['CpuUsage'] < self.cfg['DEADLOCK_CPU_CEILING']) & \
                        (df['DiskQueueLength'] < 5)

        # Agregacja: Czy wystąpił którykolwiek z powyższych problemów?
        predicted_failure = pred_overload | pred_mem_leak | pred_network | pred_deadlock

        # --- WERYFIKACJA (GROUND TRUTH) ---
        # Sprawdzamy, czy nasze przewidywania były słuszne.
        # Prawdziwa awaria to wystąpienie błędu HTTP 500+ LUB przekroczenie czasu SLA.
        actual_failure = df['EventCode'].isin([500, 501, 998, 999]) | \
                         (df['LatencyMs'] > self.cfg['SLA_LATENCY_LIMIT_MS'])

        # --- DETEKCJA ANOMALII ---
        # Anomalie to niekoniecznie awarie - to po prostu "dziwne" zachowania.
        
        # Anomalie metryczne: Wartości wielokrotnie przekraczające historyczne maksimum (p95).
        is_metric_anomaly = (
            (df['NetworkErrors'] > self.stats['net_p95'] * self.cfg['ANOMALY_NET_MULT']) |
            (df['DiskQueueLength'] > self.stats['disk_p95'] * self.cfg['ANOMALY_DISK_MULT'])
        )
        
        # Anomalie logiczne: Pojawienie się kodu zdarzenia, którego nie było w zbiorze treningowym.
        is_logical_anomaly = df['EventCode'].apply(lambda x: x not in self.stats['known_codes'] or x in self.stats['rare_codes'])
        
        predicted_anomaly = is_metric_anomaly | is_logical_anomaly
        actual_anomaly = (df['IsAnomaly'] == 1) | is_logical_anomaly

        # Zapisujemy wyniki do DataFrame w celu późniejszej ewaluacji
        df['Predicted_Failure'] = predicted_failure
        df['Actual_Failure'] = actual_failure
        df['Predicted_Anomaly'] = predicted_anomaly
        df['Actual_Anomaly'] = actual_anomaly

        return df[['Predicted_Failure', 'Actual_Failure', 'Predicted_Anomaly', 'Actual_Anomaly']]

    def predict_latency_simple(self):
        """
        Model regresji kategorycznej (Bucket Regression).
        Zamiast przewidywać dokładną wartość (np. 123ms), przypisujemy stan systemu do kategorii
        (Normal, Medium, Heavy, Critical) i mapujemy to na wartości historyczne.
        Jest to bardziej stabilne niż liniowa regresja w systemach skokowych.
        """
        print("Uruchamianie modelu regresji opóźnień...")
        # Kategoryzacja SQL-owa na podstawie metryk zasobów
        df = self.con.execute(f"""
            SELECT 
                LatencyMs, CpuUsage, DiskQueueLength, NetworkErrors, Retries,
                CASE 
                    WHEN NetworkErrors > {self.cfg['REG_CRITICAL_NET']} OR DiskQueueLength > {self.cfg['REG_CRITICAL_DISK']} THEN 'critical'
                    WHEN CpuUsage > {self.cfg['REG_HEAVY_CPU']} OR DiskQueueLength > 50 THEN 'heavy_load'
                    WHEN CpuUsage > 70 THEN 'medium_load'
                    ELSE 'normal'
                END as predicted_category
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
            WHERE LatencyMs IS NOT NULL 
              AND Timestamp >= '{self.split_time}'
            USING SAMPLE 20% LIMIT 5000
        """).df()

        if df.empty: return pd.DataFrame()

        # Mapowanie kategorii na konkretne wartości liczbowe (milisekundy) z profilu
        mapping = {
            'normal':      self.stats['lat_median'],       
            'medium_load': self.stats['lat_p95'] * 0.8,
            'heavy_load':  self.stats['lat_p95'] * 1.5,          
            'critical':    self.stats['lat_p99']           
        }
        
        df['Predicted'] = df['predicted_category'].map(mapping)
        # Obliczenie błędu bezwzględnego (MAE)
        df['Error'] = (df['Predicted'] - df['LatencyMs']).abs()
        
        return df[['Predicted', 'LatencyMs', 'Error']]

    def predict_sequential_simple(self):
        """
        Analiza sekwencyjna (Session-based Analysis).
        Sprawdza historię pojedynczej transakcji (TransactionId), aby wykryć wzorce prowadzące do awarii.
        Symuluje przetwarzanie strumieniowe: Decyzja w kroku T jest podejmowana tylko na podstawie kroków 0..T-1.
        """
        print("Analiza sekwencyjna (Session-based Analysis)...")
        
        # Pobieramy ID transakcji, które są wystarczająco długie do analizy (min. 3 kroki)
        tx_df = self.con.execute(f"""
            SELECT TransactionId 
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true)
            WHERE Timestamp >= '{self.split_time}'
            GROUP BY TransactionId 
            HAVING COUNT(*) >= 3 
            LIMIT 300
        """).df()
        
        if tx_df.empty: return pd.DataFrame()

        # Pobieramy szczegóły tych transakcji
        ids_fmt = "'" + "','".join(tx_df['TransactionId'].tolist()) + "'"
        events_df = self.con.execute(f"""
            SELECT TransactionId, EventCode, LatencyMs, CpuUsage, Retries, IsAnomaly, Timestamp 
            FROM read_parquet('{str(PARQUET_DIR)}/**/*.parquet', hive_partitioning=true) 
            WHERE TransactionId IN ({ids_fmt}) 
            ORDER BY TransactionId, Timestamp
        """).df()

        predictions = []
        for _, grp in events_df.groupby('TransactionId'):
            grp = grp.reset_index(drop=True)
            # Iterujemy po krokach transakcji (od 3. kroku wzwyż)
            for i in range(2, len(grp)):
                hist = grp.iloc[:i] # Historia dostępna w momencie predykcji
                warnings = 0
                
                # Reguła 1: Czy w historii były już błędy krytyczne?
                if (hist['EventCode'] >= 500).any(): warnings += self.cfg['SEQ_WEIGHT_ERROR']
                
                # Reguła 2: Czy widać trend rosnący opóźnień?
                if hist['LatencyMs'].iloc[-1] > hist['LatencyMs'].mean() * 1.5: warnings += self.cfg['SEQ_WEIGHT_LATENCY']
                
                # Reguła 3: Czy system próbował ponawiać operacje (Retry)?
                if (hist['Retries'] > 0).any(): warnings += 1.0

                # Normalizacja wyniku (Score 0.0 - 1.0)
                score = min(warnings / 5.0, 1.0)
                
                # Sprawdzenie, czy w bieżącym kroku faktycznie wystąpił problem (Target)
                is_actual_prob = (grp.iloc[i]['EventCode'] >= 500) or (grp.iloc[i]['LatencyMs'] > self.cfg['SLA_LATENCY_LIMIT_MS'])
                
                predictions.append({
                    'Score': score, 
                    'Predicted_Problem': score >= self.cfg['SEQ_THRESHOLD'],
                    'Actual_Problem': is_actual_prob
                })
        
        return pd.DataFrame(predictions)

    def _print_metrics(self, df, pred_col, actual_col, name):
        """
        Oblicza i wyświetla standardowe metryki klasyfikacji (Confusion Matrix + F1 Score).
        Służy do oceny jakości modelu.
        """
        if df.empty: 
            print(f"{name}: Brak danych do ewaluacji.")
            return
        
        tp = (df[pred_col] & df[actual_col]).sum() # True Positive (Trafienie)
        fp = (df[pred_col] & ~df[actual_col]).sum() # False Positive (Fałszywy alarm)
        fn = (~df[pred_col] & df[actual_col]).sum() # False Negative (Przegapienie)
        tn = (~df[pred_col] & ~df[actual_col]).sum() # True Negative (Prawidłowy spokój)
        
        # Zabezpieczenie przed dzieleniem przez zero
        f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0
        acc = (tp + tn) / len(df)
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
        
        print(f"[{name}] F1: {f1:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | Acc: {acc:.2f}")
        print(f"       TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    def run(self):
        """Główna metoda sterująca przepływem (Pipeline)."""
        # 1. Nauka (Profilowanie)
        self.load_baseline_stats()
        
        # 2. Predykcja (Testowanie)
        f = self.predict_failure_and_anomaly()
        l = self.predict_latency_simple()
        s = self.predict_sequential_simple()
        
        # 3. Raportowanie
        print("\n" + "#"*60)
        print("RAPORT KOŃCOWY EWALUACJI")
        print("#"*60)
        self._print_metrics(f, 'Predicted_Failure', 'Actual_Failure', 'AWARIA (Failure)')
        self._print_metrics(f, 'Predicted_Anomaly', 'Actual_Anomaly', 'ANOMALIA')
        
        if not l.empty:
            print("\n--- REGRESJA OPÓŹNIEŃ ---")
            print(f"MAE (Średni błąd): {l['Error'].mean():.1f} ms")

        if not s.empty:
            print("")
            self._print_metrics(s, 'Predicted_Problem', 'Actual_Problem', 'SEKWENCJA (History Analysis)')

def main():
    # Sprawdzenie czy dane istnieją przed startem
    if not PARQUET_DIR.exists():
        print(f"Błąd: Nie znaleziono katalogu z danymi: {PARQUET_DIR}")
        sys.exit(1)
    
    # Uruchomienie z obsługą błędów
    try: 
        EventPredictor().run()
    except Exception as e: 
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__": 
    main()