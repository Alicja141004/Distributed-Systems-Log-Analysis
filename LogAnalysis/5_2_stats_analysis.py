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

# Niezależne filtry stanów systemu
# Eventy mogą należeć do wielu stanów jednocześnie (np. SpikeWindow=true AND TrendWindow=true)
# NORMAL = eventy, które NIE należą do żadnego okna zjawiskowego
STATE_FILTERS = {
    "NORMAL":  "SpikeWindow = false AND FailureWindow = false AND TrendWindow = false AND IsAnomaly = 0",
    "SPIKE":   "SpikeWindow = true",
    "FAILURE": "FailureWindow = true",
    "TREND":   "TrendWindow = true",
    "ANOMALY": "IsAnomaly = 1",
}

# Kolejność wyświetlania (od łagodnego do krytycznego)
STATE_ORDER = ["NORMAL", "TREND", "ANOMALY", "SPIKE", "FAILURE"]


def main():
    if not PARQUET_DIR.exists():
        raise RuntimeError(f"Nie znaleziono katalogu parquet: {PARQUET_DIR}")

    print(f"5.2. Analiza statystyczna")
    print(f"Metryka: {METRIC}")
    print(f"Liczba \"kubełków\": {BINS_COUNT}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=?", [THREADS])
    con.execute("PRAGMA memory_limit=?", [f"{MEMORY_LIMIT_GB}GB"])

    # 1. Diagnostyka danych
    print("\n[1/6] Diagnostyka danych...")
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

    d = diag_df.iloc[0]
    print(f"   Total rows          : {d['total_rows']:>12,}")
    print(f"   Valid {METRIC:<14}: {d['valid_metric_rows']:>12,}")
    print(f"   FailureWindow=true  : {d['failure_windows']:>12,}")
    print(f"   TrendWindow=true    : {d['trend_windows']:>12,}")
    print(f"   SpikeWindow=true    : {d['spike_windows']:>12,}")
    print(f"   IsAnomaly=1         : {d['anomaly_flags']:>12,}")

    # 2. Zakresy - użyto MIN/MAX, aby objąć cały zakres danych (skala logarytmiczna poradzi sobie z outlierami)
    print("\n[2/6] Obliczanie zakresu (min-max)...")
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
    else:
        bins = np.linspace(min_val, max_val, BINS_COUNT + 1)

    print(f"   Skala: {'LOGARYTMICZNA (lepsza dla Latency)' if USE_LOG_SCALE else 'LINIOWA'}")

    # Przygotowanie definicji "kubełków" dla SQL (CASE WHEN...)
    case_parts = []
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        # Ostatni "kubełek" zamknięty z obu stron, żeby objąć max_val (który jest górną krawędzią ostatniego kubełka)
        if i == len(bins) - 2:
            case_parts.append(
                f"WHEN val >= {low} AND val <= {high} THEN {i+1}")
        else:
            case_parts.append(f"WHEN val >= {low} AND val < {high} THEN {i+1}")
    bucket_case_sql = "\n".join(case_parts)

    # 3. Widok bazowy + agregacja (niezależne filtry)
    print("\n[3/6] Agregacja danych w DuckDB (niezależne filtry stanów)...")

    # Widok utworzony raz, używany jest do dwóch zapytań
    con.execute(f"""
    CREATE TEMP VIEW base_view AS
    SELECT 
        {METRIC} AS val,
        SpikeWindow,
        FailureWindow,
        TrendWindow,
        IsAnomaly
    FROM read_parquet('{str(PARQUET_DIR).replace("'", "''")}/**/*.parquet', hive_partitioning=true)
    WHERE {METRIC} IS NOT NULL
    """)

    # a) Pobranie statystyk globalnych per stan
    global_parts = []
    for state, filt in STATE_FILTERS.items():
        global_parts.append(f"""
            SELECT
                '{state}'    AS SystemState,
                COUNT(*)     AS exact_total_events,
                AVG(val)     AS exact_mean,
                MEDIAN(val)  AS exact_median,
                VAR_POP(val) AS exact_variance,
                STDDEV(val)  AS exact_stddev
            FROM base_view
            WHERE {filt}
        """)
    global_stats_df = con.execute(" UNION ALL ".join(global_parts)).df().set_index("SystemState")

    # b) Pobranie histogramu per stan (liczba eventów w każdym "kubełku")
    hist_parts = []
    for state, filt in STATE_FILTERS.items():
        hist_parts.append(f"""
            SELECT
                '{state}' AS SystemState,
                CASE {bucket_case_sql} END AS bucket_idx,
                COUNT(*) AS bucket_count
            FROM base_view
            WHERE {filt}
            GROUP BY bucket_idx
        """)
    hist_stats_df = con.execute(" UNION ALL ".join(hist_parts)).df()

    # 4. Obliczanie wskaźników złożonych
    print("\n[4/6] Obliczanie entropii i balance ratio...")
    max_entropy = np.log2(BINS_COUNT)
    
    # Iterowanie po stanach, które faktycznie wystąpiły w statystykach globalnych
    results = []
    for state in global_stats_df.index:
        # Pobranie precyzyjnych statystyk z kroku 3a)
        stats = global_stats_df.loc[state]
        total_events = stats["exact_total_events"]
        mean_val = float(stats["exact_mean"])
        median_val = float(stats["exact_median"])
        var_val = float(stats["exact_variance"])
        std_val = float(stats["exact_stddev"])

        # Pobranie histogramu z kroku 3b)
        state_hist = hist_stats_df[hist_stats_df["SystemState"] == state]
        # Budowanie wektora liczebności (pewność, że są wszystkie kubełki, nawet puste)
        counts_map = dict(zip(state_hist["bucket_idx"], state_hist["bucket_count"]))
        vector = np.array([counts_map.get(i, 0) for i in range(1, BINS_COUNT + 1)])

        # Entropia Shannona & znormalizowana
        probs = vector / total_events if total_events > 0 else np.zeros(BINS_COUNT)
        ent = entropy(probs, base=2)
        norm_ent = ent / max_entropy if max_entropy > 0 else 0.0

        # Balance ratio (max / median kubełków; fallback max / min_nonzero)
        med_bin = np.median(vector)
        if med_bin > 0:
            balance = np.max(vector) / med_bin
        else:
            nonzero = vector[vector > 0]
            balance = float(np.max(vector) / np.min(nonzero)) if len(nonzero) > 1 else float("inf")

        results.append({
            "State": state,
            "Events": int(total_events),
            "Mean": mean_val,
            "Median": median_val,
            "Variance": var_val,
            "StdDev": std_val,
            "NormEntropy": norm_ent,
            "BalanceRatio": balance,
        })
    
    res_df = pd.DataFrame(results).set_index("State")
    order = [s for s in STATE_ORDER if s in res_df.index]
    res_df = res_df.loc[order]



    # 5. Wyświetlanie wyników
    print("\n[5/6] WYNIKI:")
    print()

    # Nagłówek
    print(f"  {'STAN':<10}│{'Events':>10} │{'Mean':>9} │{'Median':>8} │{'Variance':>13} │{'StdDev':>9} │{'NormEnt':>7} │{'BalRatio':>9} │ WNIOSEK")
    print("  " + "-" * 130)
    
    # Dane referencyjne NORMAL
    norm = res_df.loc["NORMAL"] if "NORMAL" in res_df.index else None
    norm_mean = norm["Mean"] if norm is not None else 1
    norm_ent = norm["NormEntropy"] if norm is not None else 0

    for state in order:
        r = res_df.loc[state]
    
        # Logika wnioskowania oparta na relacji do średniej i entropii
        if state == "NORMAL":
            interp = "Baseline (punkt odniesienia)"
        else:
            parts = []
            
            # Analiza przesunięcia - porównanie średnich
            # Jeśli średnia wzrosła > 10x -> Krytyczne, > 2x -> Istotne
            mean_ratio = r["Mean"] / norm_mean if norm_mean > 0 else 1.0
            
            if mean_ratio > 10.0:
                parts.append("Krytyczny skok opóź.")
            elif mean_ratio > 2.0:
                parts.append("Istotny wzrost opóź.")
            elif mean_ratio < 0.8:
                parts.append("Spadek średniej")
            else:
                parts.append("Średnia w normie")

            # Analiza chaosu - porównanie entropii
            # Wyższa entropia = bardziej płaski wykres (nieprzewidywalność)
            # Niższa entropia = wykres "szpilkowy" (determinizm)
            ent_diff = r["NormEntropy"] - norm_ent
            
            if ent_diff > 0.15:
                parts.append("Wzrost chaosu")
            elif ent_diff < -0.15:
                parts.append("Silna koncentracja")
            else:
                parts.append("Podobny rozkład do NORMAL")
            
            # Balance ratio - nierównomierność
            if r["BalanceRatio"] > 1000:
                parts.append(f"Ekstr. pik")
            elif r["BalanceRatio"] > 100:
                parts.append(f"Silny pik")
            
            interp = ", ".join(parts)

        print(f"  {state:<10}│{int(r['Events']):>10,} │{r['Mean']:>9.1f} │{r['Median']:>8.1f} │"
              f"{r['Variance']:>13.1f} │{r['StdDev']:>9.1f} │{r['NormEntropy']:>7.3f} │{r['BalanceRatio']:>9.1f} │ {interp}")

    print("  " + "-" * 130)
    print("  LEGENDA:")
    print("  * NormEnt (0-1) - Stopień chaosu. 0=wszystko w jednym binie, 1=jednakowo we wszystkich")
    print("  * BalRatio      - Nierównomierność rozkładu (max_bin / median_bin)")

    # 5.5. Wyświetlenie surowych liczników per bin
    print("\n[5.5/6] Surowe liczniki (events per bin):")
    pivot_counts = hist_stats_df.pivot(index="bucket_idx", columns="SystemState", values="bucket_count").fillna(0).astype(int)
    
    # Indeks + zakres (ms)
    bin_ranges = []
    for i in range(len(bins) - 1):
        bin_ranges.append(f"({i+1}) {int(bins[i])}-{int(bins[i+1])}")

    pivot_counts = pivot_counts.reindex(range(1, BINS_COUNT + 1), fill_value=0)
    pivot_counts.index = bin_ranges[:len(pivot_counts)]
    
    cols_to_show = [c for c in STATE_ORDER if c in pivot_counts.columns]
    print(pivot_counts[cols_to_show])

    # ---------------------------------------------------------

    print("\n[6/6] Generowanie wykresu porównawczego...")
    
    bucket_labels = {}
    for i in range(len(bins) - 1):
        bucket_labels[i + 1] = f"{i + 1}\n({int(bins[i])}-{int(bins[i + 1])})"

    plot_data = []
    for state in order:
        if state not in global_stats_df.index:
            continue
        total = global_stats_df.loc[state, "exact_total_events"]
        state_hist = hist_stats_df[hist_stats_df["SystemState"] == state]
        counts_map = dict(zip(state_hist["bucket_idx"], state_hist["bucket_count"]))

        for i in range(1, BINS_COUNT + 1):
            cnt = counts_map.get(i, 0)
            prob = cnt / total if total > 0 else 0
            plot_data.append({"State": state, "BucketIndex": i, "Probability": prob})

    if plot_data:
        pdf = pd.DataFrame(plot_data)
        plt.figure(figsize=(18, 9))

        sns.lineplot(
            data=pdf, x="BucketIndex", y="Probability",
            hue="State", marker="o", linewidth=2.5,
            hue_order=order,
        )

        xticks = list(range(1, BINS_COUNT + 1))
        xticklabels = [bucket_labels.get(i, str(i)) for i in xticks]
        plt.xticks(xticks, xticklabels, rotation=45, fontsize=8)

        scale_label = "logarytmiczne" if USE_LOG_SCALE else "liniowe"
        plt.title(
            f"Porównanie rozkładów {METRIC} - {scale_label} \"kubełki\" (k={BINS_COUNT})\n"
            f"Filtry niezależne -> eventy mogą należeć do wielu stanów jednocześnie",
            fontsize=14,
        )
        plt.ylabel("Prawdopodobieństwo (gęstość)", fontsize=12)
        plt.xlabel(f"Indeks \"kubełka\" (zakres {METRIC} [ms])", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Stan systemu", fontsize=11, loc="upper right")
        plt.tight_layout()

        output_img = THIS_DIR / "5_2_distribution_comparison.png"
        plt.savefig(output_img, dpi=150)
        print(f"   [OK] Wykres zapisano: {output_img.name}")
    else:
        print("   [INFO] Brak danych do wykresu.")

if __name__ == "__main__":
    main()
