import duckdb
import pandas as pd
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

THIS_DIR = Path(__file__).resolve().parent
PARQUET_DIR = THIS_DIR / "logs_expanded_parquet"
OUTPUT_IMG_DIR = THIS_DIR / "wyniki_analizy_img"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (16, 10)

def setup_dirs():
    if not PARQUET_DIR.exists():
        print(f"BŁĄD: Nie znaleziono {PARQUET_DIR}")
        sys.exit(1)
    OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Wyniki graficzne będą zapisane w: {OUTPUT_IMG_DIR}")

def save_plot(fig, filename):
    path = OUTPUT_IMG_DIR / filename
    fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Wygenerowano wykres: {path.name}")

def print_text_stats(con, parquet_query):
    """Wypisuje statystyki tekstowe do konsoli (wymagane w zadaniu 5.1)"""
    print("\n" + "="*80)
    print(" 5.1 EKSPLORACJA DANYCH - RAPORT TEKSTOWY")
    print("="*80)

    # 1. Podstawowe liczniki
    stats_sql = f"""
    SELECT 
        COUNT(*) as total_events,
        approx_count_distinct(TransactionId) as unique_tx,
        approx_count_distinct(CorrelationId) as unique_corr,
        approx_count_distinct(SourceSystem) as systems_count,
        CAST(MIN(Timestamp) AS VARCHAR) as log_start,
        CAST(MAX(Timestamp) AS VARCHAR) as log_end
    FROM {parquet_query}
    """
    stats = con.execute(stats_sql).fetchone()
    
    print(f"Zakres dat:       {stats[4]} do {stats[5]}")
    print(f"Liczba zdarzeń:   {stats[0]:,}")
    print(f"Unikalne transakcje (est): {stats[1]:,}")
    print(f"Unikalne korelacje (est):  {stats[2]:,}")
    print(f"Liczba systemów:  {stats[3]}")
    print("-" * 80)

    # 2. Rozkłady
    columns_to_analyze = ["Priority", "EventCode", "SourceSystem", "Scenario"]
    
    for col in columns_to_analyze:
        print(f"\nRozkład: {col} (Top 10)")
        sql = f"""
        SELECT {col}, COUNT(*) as Count, 
               ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {parquet_query}), 2) as Pct
        FROM {parquet_query}
        GROUP BY {col} 
        ORDER BY Count DESC
        LIMIT 10
        """
        df = con.execute(sql).df()
        # Formatowanie tabeli
        print(f"{col:<30} | {'Count':>10} | {'%':>6}")
        print("-" * 52)
        for _, row in df.iterrows():
            val = str(row[col])
            # Skracanie zbyt długich nazw
            if len(val) > 28: val = val[:25] + "..."
            print(f"{val:<30} | {row['Count']:>10,} | {row['Pct']:>6.2f}%")

    print("\n" + "="*80)


def plot_summary_dashboard(con, parquet_query):
    """Grafika 1: Podsumowanie liczbowe i rozkłady (Top 10)"""
    print("Generowanie dashboardu podsumowującego (wykres)...")
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.4, 2, 2])

    # Nagłówek
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis('off')
    ax_text.text(0.5, 0.5, "ANALIZA STRUKTURY LOGÓW (5.1)", 
                 ha='center', va='center', fontsize=24, fontweight='bold', color='#2c3e50')

    # Wykresy słupkowe
    distributions = ["Priority", "EventCode", "SourceSystem", "Scenario"]
    grid_positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
    colors = ["#3498db", "#e74c3c", "#9b59b6", "#2ecc71"]
    
    for i, col in enumerate(distributions):
        sql = f"""
        SELECT {col}, COUNT(*) as Count
        FROM {parquet_query}
        GROUP BY {col} 
        ORDER BY Count DESC
        LIMIT 10
        """
        df_dist = con.execute(sql).df()
        df_dist[col] = df_dist[col].astype(str)

        ax = fig.add_subplot(gs[grid_positions[i]])
        sns.barplot(data=df_dist, x="Count", y=col, ax=ax, color=colors[i])
        ax.set_title(f"Top 10: {col}")
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    save_plot(fig, "1_dashboard_podsumowanie.png")

def plot_histograms_dashboard(con, parquet_query):
    """Grafika 2: Histogramy z LOGARYTMICZNĄ SKALĄ"""
    print("Generowanie dashboardu histogramów...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    hist_configs = [
        ("LatencyMs", 50, 5000, "#e74c3c", True),
        ("CpuUsage", 5, 100, "#9b59b6", False),
        ("LocalQps", 25, 2000, "#2ecc71", True),
        ("DiskQueueLength", 1, 30, "#f1c40f", False)
    ]

    for i, (col, bucket, max_val, color, use_log) in enumerate(hist_configs):
        sql = f"""
        SELECT 
            FLOOR({col} / {bucket}) * {bucket} as BucketStart,
            COUNT(*) as Count
        FROM {parquet_query}
        WHERE {col} IS NOT NULL AND {col} <= {max_val}
        GROUP BY BucketStart
        ORDER BY BucketStart
        """
        df_hist = con.execute(sql).df()
        ax = axes[i]

        if not df_hist.empty:
            ax.bar(df_hist['BucketStart'], df_hist['Count'], width=bucket*0.9, color=color, align='edge', alpha=0.85)
            if use_log:
                ax.set_yscale('log')
                ax.set_title(f"{col} (Log Scale)", fontweight='bold')
            else:
                ax.set_title(f"{col}", fontweight='bold')
            ax.set_xlabel(f"{col} (Bucket={bucket})")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            
    fig.suptitle('Rozkłady metryk numerycznych', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot(fig, "2_dashboard_histogramy.png")

def run_exploration():
    setup_dirs()
    print(f"Start analizy wizualnej (5.1). Źródło: {PARQUET_DIR}")
    start_time = time.time()
    
    con = duckdb.connect()
    parquet_path_str = str(PARQUET_DIR).replace('\\', '/')
    parquet_query = f"read_parquet('{parquet_path_str}/**/*.parquet', hive_partitioning=true)"

    # 1. Statystyki tekstowe (NOWE)
    print_text_stats(con, parquet_query)

    # 2. Wykresy
    plot_summary_dashboard(con, parquet_query)
    plot_histograms_dashboard(con, parquet_query)

    elapsed = time.time() - start_time
    print(f"\nAnaliza zakończona w {elapsed:.2f} s.")

if __name__ == "__main__":
    run_exploration()