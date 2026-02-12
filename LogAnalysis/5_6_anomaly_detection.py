import duckdb
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"

SAMPLE_SIZE = 500_000
SAMPLE_SEED = 42

# Zasoby DuckDB
MEMORY_LIMIT_GB = 16
THREADS = 1


def main():
    if not PARQUET_DIR.exists():
        raise RuntimeError(f"Brak danych: {PARQUET_DIR}")

    print("5.6 Anomaly detection (Isolation Forest)")
    
    con = duckdb.connect()
    con.execute("PRAGMA threads=?", [THREADS])
    con.execute("PRAGMA memory_limit=?", [f"{MEMORY_LIMIT_GB}GB"])
    
    # 1. Wybranie kolumn numerycznych do wektora cech + flagi i kodu WYŁĄCZNIE do weryfikacji)
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

    # 2. Feature Engineering
    # Cechy interakcyjne kodują RELACJE między metrykami, co pozwala IF lepiej separować wielowymiarowe outliery niż surowe wartości
    print("[2/5] Budowanie wektora cech...")
    
    df['LogLatency'] = np.log1p(df['LatencyMs'])
    df['LogReqSize'] = np.log1p(df['RequestSizeBytes'])
    df['LogRespSize'] = np.log1p(df['ResponseSizeBytes'])
    df['LogQps'] = np.log1p(df['LocalQps'])

    eps = 1e-6
    
    # Stosunek błędów sieciowych do ruchu - dużo błędów przy niskim QPS jest bardziej podejrzane
    df['ErrorRate'] = df['NetworkErrors'] / (df['LogQps'] + eps)
    
    # # Kolejka dysku w relacji do ruchu - wysoka kolejka przy niskim QPS sugeruje problem inny niż obciążenie 
    df['DiskStress'] = df['DiskQueueLength'] / (df['LogQps'] + eps)
    
    # Ile QPS uzyskujemy z jednostki CPU - ekstremalnie niska lub wysoka wartość to sygnał anomalii
    df['CpuEfficiency'] = df['LogQps'] / (df['CpuUsage'] + eps)

   # Wysoki latency przy niskim QPS ("stall") vs wysoki latency przy wysokim QPS (obciążenie)
    df['LatencyPerQps'] = df['LogLatency'] / (df['LogQps'] + eps)

    # Stosunek rozmiaru request/response - nietypowe proporcje mogą wskazywać na outliery
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

    # 3. Modelowanie
    print("[3/5] Trenowanie modelu Isolation Forest...")
    
    # Parametry dobrane eksperymentalnie
    iso_forest = IsolationForest(
        n_estimators=300,
        max_samples=8192, 
        contamination='auto',
        random_state=42, n_jobs=-1
    )
    
    iso_forest.fit(X_scaled)
    scores = iso_forest.decision_function(X_scaled)
    df['AnomalyScore'] = scores
    
    # Kolumna ground truth (TYLKO do ewaluacji)
    df['ActualAnomaly'] = ((df['IsAnomaly'] == 1) | (df['EventCode'].isin([998, 999]))).astype(int)
    
    # 4. Dynamiczny próg
    print("[4/5] Wyznaczanie progu (dynamic separation)...")
    
    # Dynamic Separation - szukanie luki w dolnym ogonie rozkładu score'ów
    # Guardrails - alarm rate ograniczony do 0.01%–1%
    p001 = np.percentile(scores, 0.01)
    p01 = np.percentile(scores, 0.1)
    p05 = np.percentile(scores, 0.5)
    p1 = np.percentile(scores, 1.0)
    
    if p05 - p01 > 0.05:
        threshold = p01 + (p05 - p01) * 0.3
        method_name = "Dynamic Separation"
    else:
        # Brak wyraźnej separacji - fallback na percentyl 0.5% wszystkich score'ów
        threshold = np.percentile(scores, 0.5)
        method_name = "Percentile Fallback (p=0.5%)"

    # Guardrails - ograniczenie alarm rate do rozsądnego zakresu
    if threshold > p1:
        threshold = p1
        method_name += " + GuardMax(p1)"
    if threshold < p001:
        threshold = p001
        method_name += " + GuardMin(p0.01)"

    # Diagnostyka progów
    n_alarms = (scores < threshold).sum()
    alarm_pct = n_alarms / len(scores) * 100
    print(f"  p0.01={p001:.4f}  p0.1={p01:.4f}  p0.5={p05:.4f}  p1={p1:.4f}")
    print(f"  Wybrany próg: {threshold:.4f} ({method_name})")
    print(f"  Oczekiwanych alarmów: {n_alarms:,} ({alarm_pct:.2f}%)")
    
    df['PredictedAnomaly'] = np.where(df['AnomalyScore'] < threshold, 1, 0)

    # 5. Wyniki (ewaluacja)
    print("[5/5] Ewaluacja wyników...")
    
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
        fp_ratio = false_alarms / detected
        print(f"  FP/TP ratio: {fp_ratio:.1f}")
    if false_alarms == 0 and detected > 0:
        print(f"  -> Model ma idealną precyzję (brak fałszywych alarmów).")
    if recall < 0.5:
        print(f"  -> UWAGA: Recall poniżej 50% - model wykrywa tylko najsilniejsze anomalie.")
    elif recall >= 0.8:
        print(f"  -> Dobry recall - model wykrywa większość anomalii.")
    if false_alarms > 0:
        print(f"  -> FP wynikają prawdopodobnie z wykrywania zdarzeń typu 'Failure' oraz silnych 'Spike', które statystycznie są anomaliami, mimo braku etykiety 'IsAnomaly'.")

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