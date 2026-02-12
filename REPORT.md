### CZĘŚĆ 1: RAPORT TECHNICZNY (SRE/DevOps)

**Data:** 24.01.2026
**Do:** Zespół Inżynierii / DevOps
**Temat:** Analiza degradacji wydajności i awarii kaskadowej w klastrze usług

#### 1. Diagnoza Systemu

**Status:** **ZDEGRADOWANY / KRYTYCZNY**
System przetworzył **1,68 mln zdarzeń** w analizowanym okresie. Mimo że 95.7% zdarzeń to sukcesy (HTTP 200), obserwujemy wyraźny trend pogarszania się wydajności. Występuje **silna degradacja opóźnień** – w stanie *ANOMALY* percentyl 99 (P99) dla opóźnień wynosi aż **9932 ms** (vs 69 ms w normie). Wykryto liniowy trend wzrostowy opóźnień (slope ~1.7 ms/min), co sugeruje wyciek zasobów lub postępujące nasycenie. 0.5% zdarzeń (8,373) ma status `crit` (HTTP 500).

#### 2. Top 3 Problemy (Evidence-based)

1. **Nasycenie I/O w ApiGateway (Resource Exhaustion)**
* **Dowód:** W analizie RCA (Task 5.7) `ApiGateway` jest punktem zapalnym. W momencie awarii (timestamp: ...18.729) `DiskQueueLength` wynosił **93-145**, przy `CpuUsage` > 87%.
* **Skutek:** To powoduje odrzucanie żądań na wejściu (HTTP 500 na kroku 0 transakcji).


2. **Destrukcyjna propagacja z NotificationService i InventoryService**
* **Dowód:** Analiza propagacji (Task 5.5) wskazuje, że `NotificationService` (241 przypadków) oraz `InventoryService` (236 przypadków) są najczęstszymi inicjatorami kaskady błędów wpływających na `ApiGateway` (odbiorca problemu w 247 przypadkach).
* **Skutek:** Awaria usług backendowych "zatyka" bramkę wejściową (backpressure).


3. **Wzrostowy trend opóźnień (Trend Degradation)**
* **Dowód:** Task 5.4 wykrył 14 segmentów trendu ze średnim nachyleniem (slope) **~1.77 ms/min** i wysokim dopasowaniem (R2 > 0.99).
* **Wniosek:** System nie czyści zasobów poprawnie lub ruch rośnie szybciej niż skalowanie (Cluster 3 w Task 5.9 pokazuje "Extreme Latency" przy wysokim obciążeniu dysku).



#### 3. Łańcuch Przyczynowy (RCA) - Case Study: StockReservation

Na podstawie śladu (Trace ID: `ebe4...1305` z Task 5.7):

* **Trigger:** `ApiGateway` uległ awarii jako pierwszy (`step=0`, HTTP 500, Latency 655ms).
* **Przyczyna bezpośrednia:** Wyczerpanie zasobów na bramce. Metryki *Look-back* pokazują wysokie kolejki dyskowe (145) i błędy sieciowe (23) na sekundy przed crashem.
* **Ścieżka propagacji (wg Task 5.5):** Choć w tym konkretnym śladzie padł Gateway, globalnie statystyka pokazuje przepływ:
`InventoryService` / `NotificationService` → `ApiGateway` → `OrderService`
* **Wniosek:** `ApiGateway` jest "wąskim gardłem" (choke point), które nie wytrzymuje narzutu generowanego przez błędy i powtórzenia (Retries > 0 w klastrach awarii) z serwisów podrzędnych.

#### 4. Rekomendowane Środki Zaradcze

1. **Skalowanie ApiGateway:** Natychmiastowe zwiększenie zasobów dyskowych (IOPS) i CPU dla `ApiGateway` (Cluster 1 "Heavy Load" pokazuje 90% CPU przy 300 QPS).
2. **Circuit Breaker dla NotificationService:** Wdrożenie agresywnego odcinania `NotificationService`, gdy Latency przekracza 2000 ms (obecnie dobija do 2019 ms w błędach), aby nie blokować `ApiGateway`.
3. **Analiza Dysku:** Sprawdzenie logów pod kątem operacji dyskowych – klaster "Extreme Latency" (Task 5.9) koreluje z `DiskQueueLength` na poziomie 135-151.
4. **Rate Limiting:** Wdrożenie sztywniejszych limitów. Mamy 1.25% błędów 429 (Task 5.1), co oznacza, że obecne limity są przekraczane, ale system nadal próbuje przetwarzać zbyt wiele (szczególnie retry).

---

### CZĘŚĆ 2: KOMUNIKAT DLA CEO / STATUS PAGE

**Tytuł:** Aktualizacja statusu wydajności systemu
**Priorytet:** Wysoki

**1. Co się dzieje?**
Obserwujemy spowolnienie działania aplikacji oraz sporadyczne błędy podczas finalizacji transakcji. Dotyczy to głównie procesów: **Rezerwacji towaru (StockReservation)** oraz **Logowania**. Większość operacji (ponad 95%) kończy się sukcesem, ale czas oczekiwania na potwierdzenie może być wydłużony – w skrajnych przypadkach nawet do kilkunastu sekund (zamiast standardowych poniżej 1 sekundy).

**2. Dlaczego?**
Nasze systemy monitoringu wykryły tzw. "wąskie gardło" na głównej bramce wejściowej do systemu (ApiGateway). Jest to spowodowane kombinacją dwóch czynników:

* **Zwiększone obciążenie:** System obsługuje duży ruch (wykryliśmy segmenty "Heavy Load" z wysokim wykorzystaniem procesorów).
* **Problem dyskowy:** Serwery odpowiedzialne za przyjmowanie zamówień mają trudności z szybkim zapisywaniem danych na dyskach, co powoduje powstawanie kolejek ("zatorów") i spowalnia cały łańcuch usług, w tym powiadomienia i magazyn.

**3. Co robimy?**
Zespół techniczny podjął już konkretne kroki naprawcze:

* **Zwiększamy moc obliczeniową:** Dokładamy zasoby do kluczowej usługi (ApiGateway), aby udrożnić kolejkę zapytań.
* **Izolujemy problemy:** Wprowadzamy mechanizmy, które tymczasowo odciążą system od mniej krytycznych funkcji (jak np. opóźnione wysyłanie niektórych powiadomień), aby priorytetowo obsłużyć płatności i zamówienia.
* **Analizujemy trend:** Zidentyfikowaliśmy, że spowolnienie narastało liniowo w ostatnich godzinach – restartujemy najbardziej obciążone komponenty, aby natychmiast przywrócić szybkość działania.

**4. Kiedy będzie lepiej?**
Wdrożone środki powinny przynieść widoczną poprawę stabilności. Biorąc pod uwagę wykryty trend wzrostowy obciążenia, pełnej normalizacji czasów odpowiedzi spodziewamy się po wyskalowaniu infrastruktury dyskowej. Będziemy informować o postępach na bieżąco.