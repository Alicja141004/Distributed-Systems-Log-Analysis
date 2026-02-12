"""
5.10 - Generowanie human-friendly raportu z wyników analiz 5.1-5.9.

Zbiera outputy z pliku 5_10_collected_outputs.txt, łączy z promptem i zapisuje do pliku gotowego do wklejenia w AI.
"""

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
OUTPUTS_FILE = THIS_DIR / "5_10_collected_outputs.txt"
PROMPT_OUTPUT = THIS_DIR / "5_10_prompt_for_AI.txt"

PROMPT_TEMPLATE = """Jesteś Senior SRE (Site Reliability Engineer) oraz Managerem Komunikacji w firmie, która prowadzi rozproszony system składający się z kilku różnych usług: AuthService, ApiGateway, OrderService, PaymentService, NotificationService, InventoryService, ReportingService, etc.

Otrzymujesz poniżej surowe wyniki z 9 zadań analizy logów (5.1 - 5.9: eksploracja danych, analiza statystyczna, spike detection, trend detection, failure detection, anomaly detection, root cause analysis, event prediction, clustering). Dane pochodzą z syntetycznego generatora, ale traktuj je jak produkcyjne.

## TWOJE ZADANIE

Na podstawie WYŁĄCZNIE danych poniżej, przygotuj przygotuj odpowiedź składającą się z DWÓCH CZĘŚCI. Całość max 2 strony A4.

---

### CZĘŚĆ 1: KRÓTKI RAPORT TECHNICZNY (max 40% długości odpowiedzi)
Cel: Dla zespołu IT/DevOps.
Format: Markdown.

Struktura:
1. **Diagnoza (3-4 zdania):** Executive summary - stan systemu (zdrowy / zdegradowany / krytyczny). Skala problemów. Eksplorację danych oraz analizę statystyczną potraktuj bardziej jako sugestię/"ground truth".
2. **Top 3 problemy:** Każdy z: Co? Gdzie (serwis)? Dowód liczbowy (cytuj konkretne wartości z outputów).
3. **Łańcuch przyczynowy (RCA):** Na podstawie zadania 5.7 (Trace) i 5.5 (Propagacja): Który system zawiódł pierwszy? Jak awaria przepłynęła przez system (ścieżka)?
4. **Środki zaradcze (3-5 punktów):** Konkretne akcje powiązane z wykrytymi problemami.

### CZĘŚĆ 2: NIE-TECHNICZNY KOMUNIKAT DLA UŻYTKOWNIKA KOŃCOWEGO / CEO (max 60% długości odpowiedzi)
Cel: Wyjaśnić CEO/"Kowalskiemu", co się dzieje, bez używania żargonu (nie pisz: "Isolation Forest", "HTTP 500", "Latency P99", "F1-score", "threshold", etc.).
Format: Wpis na Status Page / powiadomienie w aplikacji.
Styl: Empatyczny, transparentny, spokojny.

Struktura:
1. **Co się dzieje?** (np. "Składanie zamówień może trwać dłużej niż zwykle, etc.")
2. **Dlaczego?** (uproszczona przyczyna, np. "Zwiększone obciążenie systemu płatności", etc.)
3. **Co robimy?** (konkretne środki zaradcze (w jaki DOKŁADNIE obszar systemu inwestujemy) przetłumaczone na ludzki język)
4. **Kiedy będzie lepiej?** (na podstawie trendów i predykcji z danych, jeśli możliwe)

## ZASADY
- Bądź KONKRETNY — cytuj liczby z danych (czasy odpowiedzi, % błędów, liczby zdarzeń, etc.)
- NIE wymyślaj danych, których nie ma w outputach
- Jeśli czegoś nie da się wywnioskować z danych, napisz to wprost
- Pisz po polsku
- Format: Markdown

## WSKAZÓWKI — NA CO GŁÓWNIE ZWRÓCIĆ UWAGĘ W DANYCH:
- Task 5.2: tabela WYNIKI ze statystykami per stan (NORMAL/SPIKE/FAILURE/ANOMALY) — kluczowe metryki
- Task 5.3: tabela PRF1 — skuteczność wykrywania spike'ów
- Task 5.4: wykryte segmenty trendu, slope, Precision/Recall
- Task 5.5: progi failure, PRF1 per label, propagacja między systemami
- Task 5.6: Precision/Recall/F1 anomaly detection, confusion matrix
- Task 5.7: ścieżka przejścia systemów, root cause event, look-back
- Task 5.8: F1 predykcji awarii, MAE regresji, F1 predykcji sekwencyjnej
- Task 5.9: interpretacja klastrów K-Means (NORMAL / NETWORK ISSUES / CRITICAL FAILURE)
- Jeżeli w tasku podano logiczne wnioski wyjaśniające, weź je pod uwagę.

## DANE WEJŚCIOWE (z analizy logów - outputy zadań 5.1-5.9):

```
{outputs}
```
"""


def main():
    if not OUTPUTS_FILE.exists():
        print(f"BŁĄD: Nie znaleziono {OUTPUTS_FILE}")
        print("Uruchom najpierw: bash run_tasks.sh")
        sys.exit(1)

    outputs = OUTPUTS_FILE.read_text(encoding="utf-8", errors="replace")

    # Ograniczenie rozmiaru (AI ma limit kontekstu)
    MAX_CHARS = 100_000
    if len(outputs) > MAX_CHARS:
        print(
            f"  UWAGA: Output ma {len(outputs):,} znaków, obcinanie do {MAX_CHARS:,}")
        outputs = outputs[:MAX_CHARS] + \
            "\n\n[...OBCIĘTO — za dużo danych...]\n"

    prompt = PROMPT_TEMPLATE.format(outputs=outputs)

    # Zapis do pliku
    PROMPT_OUTPUT.write_text(prompt, encoding="utf-8")
    print(f"Prompt zapisany do: {PROMPT_OUTPUT.name}")
    print(f"  Rozmiar: {len(prompt):,} znaków (~{len(prompt)//4:,} tokenów)")

    print(f"\nNastępny krok: wklej zawartość {PROMPT_OUTPUT.name} do AI.")


if __name__ == "__main__":
    main()
