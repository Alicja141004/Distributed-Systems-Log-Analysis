from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

THIS_DIR = Path(__file__).resolve().parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet" # LogAnalysis/logs_expanded_parquet

METRIC = "LatencyMs"
USE_LOG_SCALE = True # Log-binning jest LEPSZY dla Latency

# Dla 1 GB (N ~ 1,68 mln) - reguła Sturgesa daje k ~ 21.68 -> 22
# Dla 100 GB (N ~ 168 mln) - reguła Sturgesa daje k ~ 28.32 -> 29
BINS_COUNT = 25

# Zasoby DuckDB
THREADS = 8
MEMORY_LIMIT_GB = 16


def main():
    if not PARQUET_DIR.exists():
        raise RuntimeError(f"Nie znaleziono katalogu parquet: {PARQUET_DIR}")

    print(f"Analiza statystyczna 5.2")
    print(f"Metryka: {METRIC}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=?", [THREADS])
    con.execute("PRAGMA memory_limit=?", [f"{MEMORY_LIMIT_GB}GB"])

    # 1. Diagnostyka danych
    print("\n[0/5] Diagnostyka danych...")
    diag_df = con.execute(f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT({METRIC}) as valid_metric_rows,
            COUNT(CASE WHEN FailureWindow = true THEN 1 END) as failure_windows,
            COUNT(CASE WHEN TrendWindow = true THEN 1 END) as trend_windows,
            COUNT(CASE WHEN SpikeWindow = true THEN 1 END) as spike_windows,
            COUNT(CASE WHEN IsAnomaly = 1 THEN 1 END) as anomaly_flags
        FROM read_parquet('{str(PARQUET_DIR).replace("'", "''")}/**/*.parquet', hive_partitioning=true)
    """).df()

    print(f"   Total rows: {diag_df['total_rows'][0]}")
    print(f"   Valid {METRIC} rows: {diag_df['valid_metric_rows'][0]}")
    print(f"   Failure Windows rows: {diag_df['failure_windows'][0]}")
    print(f"   Trend Windows rows: {diag_df['trend_windows'][0]}")
    print(f"   Spike Windows rows: {diag_df['spike_windows'][0]}")
    print(f"   Anomaly Flags rows: {diag_df['anomaly_flags'][0]}")

    # 2. Zakresy - użyto MIN/MAX, aby objąć cały zakres danych (skala logarytmiczna poradzi sobie z outlierami)
    print("\n[1/5] Obliczanie zakresu (min-max)...")
    bounds = con.execute(f"""
        SELECT MIN({METRIC}), MAX({METRIC})
        FROM read_parquet('{str(PARQUET_DIR).replace("'", "''")}/**/*.parquet', hive_partitioning=true)
        WHERE {METRIC} IS NOT NULL
    """).fetchone()

    min_val, max_val = bounds
    if min_val is None: min_val = 1
    if max_val is None: max_val = 1000

    # Zabezpieczenie dla logarytmu (log(0) -> error)
    if min_val <= 0: min_val = 0.001 if USE_LOG_SCALE else 0

    print(f"   Zakres: {min_val} - {max_val} ms")

    # Generowanie krawędzi "kubełków" (logarytmicznie lub liniowo)
    if USE_LOG_SCALE:
        # np. logspace od log10(1) do log10(10000)
        bins = np.logspace(np.log10(min_val), np.log10(max_val), BINS_COUNT + 1)
        print(f"   Użyto skali LOGARYTMICZNEJ (lepsza dla Latency)")
    else:
        bins = np.linspace(min_val, max_val, BINS_COUNT + 1)
        print(f"   Użyto skali LINIOWEJ")

    # Przygotowanie definicji "kubełków" dla SQL (CASE WHEN...)
    case_parts = []
    for i in range(len(bins) - 1):
        low = bins[i]
        high = bins[i+1]
        # Ostatni "kubełek" jest zamknięty z obu stron, żeby objąć max_val (który jest górną krawędzią ostatniego kubełka)
        if i == len(bins) - 2:
            case_parts.append(
                f"WHEN val >= {low} AND val <= {high} THEN {i+1}")
        else:
            case_parts.append(f"WHEN val >= {low} AND val < {high} THEN {i+1}")

    bucket_case_sql = "\n".join(case_parts)

    # 3. Definicja stanów (widok tymczasowy z kolumną SystemState) i agregacja
    print("\n[2/5] Agregacja danych w DuckDB...")

    # Widok utworzony raz, używany jest do dwóch zapytań
    con.execute(f"""
    CREATE TEMP VIEW analysis_view AS
    SELECT 
        {METRIC} as val,
        CASE 
            WHEN FailureWindow = true THEN 'FAILURE'
            WHEN SpikeWindow = true THEN 'SPIKE'
            WHEN TrendWindow = true THEN 'TREND'
            WHEN IsAnomaly = 1 THEN 'ANOMALY'
            ELSE 'NORMAL'
        END as SystemState
    FROM read_parquet('{str(PARQUET_DIR).replace("'", "''")}/**/*.parquet', hive_partitioning=true)
    WHERE {METRIC} IS NOT NULL
    """)

    # a) Pobranie statystyk GLOBALNYCH (prawdziwa średnia i wariancja)
    # DuckDB policzy to precyzyjnie na surowych danych, bez wpływu "kubełków"
    global_stats_df = con.execute("""
        SELECT 
            SystemState,
            COUNT(*) as exact_total_events,
            AVG(val) as exact_mean,
            VAR_POP(val) as exact_variance,
            STDDEV(val) as exact_stddev
        FROM analysis_view
        GROUP BY SystemState
    """).df().set_index("SystemState")

    # b) Pobranie HISTOGRAMU (tylko liczniki do entropii) - z podziałem na "kubełki"
    hist_stats_df = con.execute(f"""
        SELECT
            SystemState,
            CASE {bucket_case_sql} END as bucket_idx,
            COUNT(*) as bucket_count
        FROM analysis_view
        GROUP BY SystemState, bucket_idx
    """).df()

    # 4. Łączenie wyników i obliczanie wskaźników złożonych (entropia, balance ratio)
    print("\n[3/5] Obliczanie entropii i balance ratio...")
    results = []

    # Iterowanie po stanach, które faktycznie wystąpiły w statystykach globalnych
    for state in global_stats_df.index:
        # Pobranie precyzyjnych statystyk z kroku a)
        exact_stats = global_stats_df.loc[state]
        total_events = exact_stats['exact_total_events']
        mean_val = exact_stats['exact_mean']
        std_val = exact_stats['exact_stddev']

        # b) Pobranie histogramu z kroku b)
        state_hist = hist_stats_df[hist_stats_df['SystemState'] == state]

        # Budowanie wektora liczebności (pewność, że są wszystkie kubełki, nawet puste)
        counts_map = dict(
            zip(state_hist['bucket_idx'], state_hist['bucket_count']))
        vector = [counts_map.get(i, 0) for i in range(1, BINS_COUNT + 1)]

        # Liczenie wskaźników złożonych (entropia, balance ratio)
        probs = np.array(vector) / total_events
        ent = entropy(probs, base=2)

        med = np.median(vector)
        ratio = np.max(vector) / med if med > 0 else np.max(vector)

        # Indeks najczęstszego "kubełka"
        mode_idx = np.argmax(vector) + 1  # +1 bo indeksy tablicy od 0, a kubełki od 1

        results.append({
            "State": state,
            "Events": int(total_events),
            "Mean": round(mean_val, 2),
            "StdDev": round(std_val, 2),
            "Entropy": round(ent, 4),
            "ModeBin": int(mode_idx),
            "BalanceRatio": round(ratio, 2)
        })
    
    res_df = pd.DataFrame(results).set_index("State")

    # Sortowanie logiczne w tabeli wynikowej - FAILURE > SPIKE > TREND > ANOMALY > NORMAL
    order = ["NORMAL", "ANOMALY", "TREND", "SPIKE", "FAILURE"]
    order = [o for o in order if o in res_df.index]
    res_df = res_df.loc[order]

    print("\n[4/5] WYNIKI:")
    print("-" * 90)
    print(res_df.to_string())
    print("-" * 90)

    print("\n[5/5] KLUCZOWE RELACJE STATYSTYCZNE:")
    print("-" * 90)

    if "NORMAL" in res_df.index:
        norm_mean = res_df.loc["NORMAL", "Mean"]
        norm_std = res_df.loc["NORMAL", "StdDev"]
        norm_cv = norm_std / norm_mean if norm_mean > 0 else 0

        print(f"{'STATE':<10} | {'Z-SCORE (Odległość)':<20} | {'CV (Zmienność)':<15} | {'WNIOSEK STATYSTYCZNY'}")
        print("-" * 90)

        for state in res_df.index:
            if state == "NORMAL":
                print(f"{state:<10} | {'0.00 sigma (REF)':<20} | {norm_cv:.2f} {'(Niska)' if norm_cv < 0.5 else '(Wysoka)' :<8}   | Punkt odniesienia (baseline)")
                continue

            # Dane stanu
            curr_mean = res_df.loc[state, "Mean"]
            curr_std = res_df.loc[state, "StdDev"]
            
            # Z-Score (ile razy odchylenie standardowe normy mieści się w różnicy średnich)
            # (Mean_State - Mean_Normal) / Std_Normal
            z_score = (curr_mean - norm_mean) / norm_std if norm_std > 0 else 0
            
            # Coefficient of Variation (CV) - współczynnik zmienności
            # StdDev / Mean. Mówi o "stabilności" w ramach tego stanu.
            cv = curr_std / curr_mean if curr_mean > 0 else 0
            
            interpretation = []
            if z_score > 3: interpretation.append("Istotne przesunięcie")
            else: interpretation.append("Blisko normy")
            
            if cv > norm_cv * 2: interpretation.append("Wzrost chaosu")
            elif cv < norm_cv * 0.5: interpretation.append("Wysoka koncentracja")
            else: interpretation.append("Podobna dynamika")

            print(f"{state:<10} | {z_score:>6.2f} sigma         | {cv:.2f}            | {', '.join(interpretation)}")

        print("-" * 90)
        print("LEGENDA:")
        print(" * Z-Score - mówi jak bardzo 'nienormalna' jest średnia tego stanu (względem zmienności normy).")
        print(" * CV (Std/Mean) - mówi czy stan jest stabilny (niskie CV) czy rozchwiany (wysokie CV).")
    else:
        print("[INFO] Brak stanu NORMAL do wyliczenia relacji.")
        
    # ---------------------------------------------------------

    print("\n[INFO] Generowanie wykresu porównawczego...")
    
    # Przygotowanie etykiet osi X (zakresy ms)
    bucket_labels = {}
    for i in range(len(bins) - 1):
        low = bins[i]
        high = bins[i+1]
        bucket_idx = i + 1
        label = f"{bucket_idx}\n({int(low)}-{int(high)})"
        bucket_labels[bucket_idx] = label
        
    # Przygotowanie danych do wykresu
    plot_data = []
    target_states = ["NORMAL", "FAILURE", "TREND", "SPIKE", "ANOMALY"] 
    
    for state in target_states:
        if state not in global_stats_df.index: continue
            
        total = global_stats_df.loc[state, 'exact_total_events']
        state_hist = hist_stats_df[hist_stats_df['SystemState'] == state]
        
        counts_map = dict(zip(state_hist['bucket_idx'], state_hist['bucket_count']))
        
        for i in range(1, BINS_COUNT + 1):
            cnt = counts_map.get(i, 0)
            # Normalizacja do gęstości prawdopodobieństwa (suma = 1)
            prob = cnt / total if total > 0 else 0
            plot_data.append({
                "State": state,
                "BucketIndex": i,
                "Probability": prob
            })

    # Rysowanie wykresu
    if plot_data:
        pdf = pd.DataFrame(plot_data)
        
        plt.figure(figsize=(16, 8))
            
        sns.lineplot(data=pdf, x="BucketIndex", y="Probability", hue="State", marker="o", linewidth=2.5)
            
        xticks = list(range(1, BINS_COUNT + 1))
        xticklabels = [bucket_labels.get(i, str(i)) for i in xticks]
            
        plt.xticks(xticks, xticklabels, rotation=45, fontsize=9)
            
        plt.title(f"Porównanie rozkładów ({METRIC}) - logarytmiczne \"kubełki\"", fontsize=16)
        plt.ylabel("Prawdopodobieństwo (gęstość)", fontsize=12)
        plt.xlabel("Indeks \"kubełka\" & (zakres ms)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Stan systemu", fontsize=11)
            
        plt.tight_layout()
            
        output_img = THIS_DIR / "5_2_distribution_comparison.png"
        plt.savefig(output_img)
        print(f"   [OK] Wykres zapisano jako: {output_img.name}")
            
    else:
        print("   [INFO] Brak danych do wykresu.")

if __name__ == "__main__":
    main()
