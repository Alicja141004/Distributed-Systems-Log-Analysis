#!/bin/bash

set -euo pipefail

REPORT_FILE="$(dirname "$0")/5_10_collected_outputs.txt"

scripts=(
  "5_1_exploration.py"
  "5_2_stats_analysis.py"
  "5_3_spike_detection.py"
  "5_4_trend_detection.py"
  "5_5_failure_detection.py"
  "5_6_anomaly_detection.py"
  "5_7_root_cause_analysis.py"
  "5_8_event_prediction.py"
  "5_9_kmeans.py"
)

export PYTHONIOENCODING=utf-8

# Wyczyść / stwórz plik raportu
> "$REPORT_FILE"

failed=()

for s in "${scripts[@]}"; do
  task_name="${s%.py}"
  echo "==> Running $s"
  
  {
    echo "================================================================"
    echo "TASK: $task_name"
    echo "================================================================"
    # Kontynuuj nawet gdy skrypt się wywali (|| true)
    python "$s" 2>&1 || echo "[BŁĄD] $s zakończył się z kodem $?"
    echo ""
    echo ""
  } | tee -a "$REPORT_FILE"

  # Sprawdź PIPESTATUS (exit code pythona, nie tee)
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    failed+=("$s")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "" | tee -a "$REPORT_FILE"
  echo "UWAGA: Następujące skrypty zakończyły się błędem:" | tee -a "$REPORT_FILE"
  for f in "${failed[@]}"; do
    echo "  - $f" | tee -a "$REPORT_FILE"
  done
fi

echo "================================================================"
echo "Wszystkie outputy zapisane w: $REPORT_FILE"

echo ""
echo "==> Generowanie prompta 5.10..."
python "5_10_report.py"