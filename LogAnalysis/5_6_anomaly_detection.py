import duckdb
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet" # LogAnalysis/logs_expanded_parquet

SAMPLE_SIZE = 500_000
SAMPLE_SEED = 42

# Zasoby DuckDB
MEMORY_LIMIT_GB = 16
THREADS = 1 # Gwarancja powtarzalności przy REPEATABLE


def main():
    if not PARQUET_DIR.exists():
        raise RuntimeError(f"Brak danych: {PARQUET_DIR}")

    print("5.6 Anomaly detection (Isolation Forest)")
    
    con = duckdb.connect()
    con.execute("PRAGMA threads=?", [THREADS])
    con.execute("PRAGMA memory_limit=?", [f"{MEMORY_LIMIT_GB}GB"])
    
    # 1. Pobranie danych (wybranie kolumn numerycznych do wektora cech + flagi i kodu WYŁĄCZNIE do weryfikacji)
    # Pobranie próbki losowej, aby nie zapchać RAM przy uczeniu modelu, ale wystarczająco dużą, by złapać rzadkie anomalie
    print(f"[1/5] Pobieranie danych (próbka {SAMPLE_SIZE} wierszy)...")

    df = con.execute(f"""
        SELECT 
            LatencyMs, 
            CpuUsage, 
            DiskQueueLength, 
            NetworkErrors, 
            LocalQps, 
            RequestSizeBytes, 
            ResponseSizeBytes,
            
            -- Kolumny ewaluacyjne (NIE wchodzą do modelu)
            IsAnomaly,
            EventCode
        FROM read_parquet('{str(PARQUET_DIR).replace("'", "''")}/**/*.parquet', hive_partitioning=true)
        USING SAMPLE reservoir({SAMPLE_SIZE} ROWS)
        REPEATABLE({SAMPLE_SEED})
    """).df()

    n_actual = ((df['IsAnomaly'] == 1) | (df['EventCode'].isin([998, 999]))).sum()
    print(f"  Wierszy: {len(df):,}  |  Rzeczywiste anomalie w próbce: {n_actual}")

    # 2. Feature Engineering (logiczne podejście - wykryć anomalie PO CHARAKTERYSTYCE wielowymiarowej, nie po etykietach)
    # Isolation Forest znajdzie punkty izolowane w przestrzeni cech
    print("[2/5] Budowanie wektora cech...")
    
    # A. Logarytmy dla rozkładów potęgowych
    df['LogLatency'] = np.log1p(df['LatencyMs'])
    df['LogReqSize'] = np.log1p(df['RequestSizeBytes'])
    df['LogRespSize'] = np.log1p(df['ResponseSizeBytes'])
    df['LogQps'] = np.log1p(df['LocalQps'])

    # B. Cechy interakcyjne - kodują RELACJE między metrykami
    # Celem jest odróżnić różne typy odstających eventów
    eps = 1e-6
    
    # 1. ErrorRate - czy błędy sieciowe są proporcjonalne do ruchu?
    # Event z dużą liczbą błędów przy niskim ruchu jest bardziej podejrzany
    df['ErrorRate'] = df['NetworkErrors'] / (df['LogQps'] + eps)
    
    # 2. DiskStress - kolejka dysku w relacji do ruchu
    # Duża kolejka przy niskim QPS sugeruje problem wydajnościowy inny niż zwykłe obciążenie
    df['DiskStress'] = df['DiskQueueLength'] / (df['LogQps'] + eps)
    
    # 3. CpuEfficiency - ile QPS uzyskujemy z jednostki CPU?
    # Event z ekstremalnym CPU (bardzo niskim lub bardzo wysokim) przy nietypowym QPS będzie miał odstającą wartość tej cechy
    df['CpuEfficiency'] = df['LogQps'] / (df['CpuUsage'] + eps)

    # 4. LatencyPerQps - wysoki latency przy niskim QPS ("stall") vs wysoki latency przy wysokim QPS (obciążenie)
    # Ta cecha separuje te dwa zjawiska
    df['LatencyPerQps'] = df['LogLatency'] / (df['LogQps'] + eps)

    # 5. Anomalie mają ekstremalny RequestSize (1-100 LUB 50k-200k)
    # Stosunek req/resp - anomalie mają nietypowe proporcje
    df['SizeRatio'] = df['LogReqSize'] / (df['LogRespSize'] + eps)

    # Wybór ostatecznych cech
    features = [
        'LogLatency',
        'CpuUsage',
        'LogQps',
        'ErrorRate',
        'DiskStress',
        'CpuEfficiency',
        'LatencyPerQps',
        'SizeRatio',
    ]
    
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Modelowanie (najlepsze parametry z eksperymentów)
    print("[3/5] Trenowanie modelu Isolation Forest...")
    
    # Parametry dobrane eksperymentalnie (duża próbka kluczowa dla jakości)
    iso_forest = IsolationForest(
        n_estimators=300,
        max_samples=8192, 
        contamination='auto',
        random_state=42, n_jobs=-1
    )
    
    iso_forest.fit(X_scaled)
    scores = iso_forest.decision_function(X_scaled)
    df['AnomalyScore'] = scores
    
    # 4. Dynamiczny próg
    print("[4/5] Wyznaczanie progu (dynamic separation)...")
    
    # Szukanie "luki" w ogonie rozkładu score'ów
    # Jeśli anomalie są dobrze izolowane, między nimi a resztą będzie przerwa w score'ach
    p01 = np.percentile(scores, 0.1) 
    p05 = np.percentile(scores, 0.5)
    
    # Jeśli jest wyraźna separacja, użyty zostanie dynamiczny próg. Jeśli nie, fallback na 0.2%.
    if p05 - p01 > 0.05:
        threshold = p01 + (p05 - p01) * 0.3
        method_name = "Dynamic Separation"
    else:
        threshold = np.percentile(scores, 0.2)
        method_name = "Percentile 0.2% (fallback)"

    df['PredictedAnomaly'] = np.where(df['AnomalyScore'] < threshold, 1, 0)

    # 5. Wyniki (ewaluacja)
    print("[5/5] Ewaluacja wyników...")
    
    df['ActualAnomaly'] = ((df['IsAnomaly'] == 1) | (df['EventCode'].isin([998, 999]))).astype(int)
    y_true = df['ActualAnomaly']
    y_pred = df['PredictedAnomaly']
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "=" * 60)
    print("WYNIKI ANOMALY DETECTION - Isolation Forest (nienadzorowane)")
    print("=" * 60)
    print(f"Metoda progu : {method_name} (wartość: {threshold:.4f})")
    print(f"Cechy ({len(features)}): {', '.join(features)}")
    print("-" * 60)
    print(f"PRECISION : {precision:.4f}")
    print(f"RECALL    : {recall:.4f}")
    print(f"F1-SCORE  : {f1:.4f}")
    print("-" * 60)
    print(f"Confusion matrix:")
    print(f"  TN={cm[0][0]:>7,}  FP={cm[0][1]:>5,}")
    print(f"  FN={cm[1][0]:>7,}  TP={cm[1][1]:>5,}")

    # Podsumowanie dla człowieka
    total_anomalies = int(y_true.sum())
    detected = int(cm[1][1])
    false_alarms = int(cm[0][1])
    total_alarms = int(y_pred.sum())
    
    print(f"\nPODSUMOWANIE:")
    print(f"  Wykryto {detected} z {total_anomalies} anomalii ({recall:.1%}).")
    print(f"  Alarmów łącznie: {total_alarms}, z czego {false_alarms} fałszywych.")
    if detected > 0:
        print(f"  FP/TP ratio: {false_alarms / detected:.1f}")
        print(f"  -> Oznacza to, że na każdą prawdziwą anomalię przypada {false_alarms / detected:.1f} fałszywych alarmów.")
        print(f"  -> Metoda jest skuteczna w wykrywaniu anomalii, ale generuje relatywnie wysoką liczbę fałszywych alarmów, co jest typowe dla nienadzorowanych metod detekcji anomalii. Wysoka liczba FP wynika prawdopodobnie z wykrywania zdarzeń typu 'Failure' oraz silnych 'Spike', które statystycznie są anomaliami, mimo braku etykiety 'IsAnomaly'.")

    # 6. Minimalna walidacja
    print("\n" + "=" * 60)
    print("WALIDACJA - szybkie porównanie z sekcją 5.2.")
    print("=" * 60)

    tp_df = df[(df['PredictedAnomaly'] == 1) & (df['ActualAnomaly'] == 1)]
    lat = tp_df['LatencyMs']

    if len(lat) == 0:
        print("Brak prawdziwych anomalii wykrytych przez model.")
    else:
        print(f"  Wykryte prawdziwe anomalie (TP): N={len(lat)}")
        print(f"  LatencyMs (min/median/max): "f"{lat.min():.0f} / {lat.median():.0f} / {lat.max():.0f}")
        print("  -> Porównaj z 5.2. (min/median/max LatencyMs dla ANOMALY).")

if __name__ == "__main__":
    main()