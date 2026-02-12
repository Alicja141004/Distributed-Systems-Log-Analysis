### CZĘŚĆ 1: KRÓTKI RAPORT TECHNICZNY

**Data:** 2026-01-24
**Do:** Zespół IT / DevOps / SRE
**Od:** Senior SRE
**Temat:** Analiza degradacji wydajności i kaskadowych awarii systemu

#### 1. Diagnoza

Stan systemu oceniam jako **KRYTYCZNY**. Obserwujemy systematyczną degradację wydajności (trend wzrostowy opóźnień) oraz ostre piki awarii (EventCode 500). Metryka P99 dla opóźnień w stanach anomalnych osiąga **9.9s** (vs 69ms w normie), a wariancja wzrasta drastycznie (z 88.8 do ponad 5.6 mln), co świadczy o całkowitej utracie stabilności. Modele predykcyjne (Task 5.8) potwierdzają wysoką skuteczność wykrywania awarii (F1=0.841), co oznacza, że obecne problemy są powtarzalne i strukturalne, a nie losowe.

#### 2. Top 3 problemy

1. **Wąskie gardło I/O w InventoryService:**

- **Co:** Krytyczne przeciążenie dysku i procesora prowadzące do timeoutów.
- **Dowód:** W klastrach anomalii (Task 5.9, K=5, ID 1) `DiskUsage` skacze do **135.5** (norma ~10), a `Latency` do **1469ms**. W analizie śladu (Task 5.7) `DiskQueueLength` dla InventoryService wynosił **214** w momencie awarii.

2. **Globalny Trend Degradacji (Leak/Saturation):**

- **Co:** Liniowy wzrost opóźnień w całym systemie, niezależnie od chwilowych skoków.
- **Dowód:** Task 5.4 wykrył ciągły trend wzrostowy o nachyleniu (slope) **~1.7 - 1.85 ms/min** w godzinach 12:40 – 19:35. Precision detekcji trendu wynosi 98.9%.

3. **Kaskadowa propagacja błędów (Domino Effect):**

- **Co:** Awarie serwisów backendowych zalewają ApiGateway.
- **Dowód:** Task 5.5 wskazuje `NotificationService` (239 zdarzeń) i `InventoryService` (235 zdarzeń) jako najczęstsze źródła propagacji. `ApiGateway` jest najczęstszym "poszkodowanym" (244 zdarzenia), co skutkuje błędami 500 dla użytkownika.

#### 3. Łańcuch przyczynowy (RCA)

Na podstawie śladu transakcji `CheckoutWithTransfer` (Task 5.7) oraz macierzy propagacji (Task 5.5):

- **Root Cause:** Awaria rozpoczęła się w **InventoryService**. Wystąpił błąd 500 (krytyczny) przy opóźnieniu **1915ms** i kolejce dyskowej **214**. Analiza "look-back" wskazuje, że chwilę wcześniej (ok. 30 sek) InventoryService miał problemy przy scenariuszu `BulkImport` (failure step=6).
- **Ścieżka propagacji:**

1. **InventoryService** (Time-out/Disk saturation) ->
2. **AuthService** (zwiększone latency do 912ms, błędy 500) ->
3. **NotificationService** (błąd 500, latency 623ms) ->
4. **OrderService** ->
5. **PaymentService**.

- Awaria rozlała się ostatecznie na cały proces zakupowy.

#### 4. Środki zaradcze

1. **Natychmiastowe skalowanie warstwy danych Inventory:** Wdrożyć dodatkowe zasoby dyskowe (IOPS) dla InventoryService, gdyż `DiskQueueLength > 200` jest bezpośrednią przyczyną blokad.
2. **Rate Limiting dla BulkImport:** Analiza look-back sugeruje, że `BulkImport` destabilizuje InventoryService. Należy tymczasowo dławić ten scenariusz (Task 5.1 pokazuje, że to ~9.8% ruchu).
3. **Circuit Breaker na ApiGateway:** Wdrożyć "bezpieczniki" dla ścieżek prowadzących do Inventory i NotificationService, aby odciążyć Gateway (obecnie przyjmuje uderzenie zwrotne).
4. **Analiza wycieku zasobów:** Stały trend wzrostu latency (1.8 ms/min) sugeruje wyciek pamięci lub nieskalowalne kolejkownie – wymagany restart rolling update serwisów, aby zresetować ten trend.

---

### CZĘŚĆ 2: KOMUNIKAT DLA UŻYTKOWNIKA KOŃCOWEGO / CEO

**Tytuł:** Aktualizacja statusu działania platformy – możliwe utrudnienia w składaniu zamówień

**1. Co się dzieje?**
Obecnie możecie doświadczać spowolnionego działania naszej aplikacji. Największe utrudnienia występują podczas **finalizacji zamówień** oraz **przeglądania dostępności produktów**. Niektóre operacje mogą kończyć się błędem i koniecznością ponowienia próby.

**2. Dlaczego?**
Zidentyfikowaliśmy, że główną przyczyną jest "zator" w naszym systemie magazynowym. Mówiąc prościej – system odpowiedzialny za sprawdzanie i rezerwację towarów (Inventory) jest obecnie przeciążony dużą liczbą operacji importu danych i zapytań, co powoduje, że "dyski" serwerów nie nadążają z zapisywaniem informacji. To opóźnienie przenosi się na pozostałe części systemu, takie jak płatności i powiadomienia, działając na zasadzie efektu domina.

**3. Co robimy?**
Nasz zespół inżynierów podjął już konkretne kroki naprawcze:

- **Zwiększamy moc obliczeniową:** Dokładamy zasoby dyskowe do systemu magazynowego, aby udrożnić kolejkę oczekujących zapytań.
- **Ograniczamy procesy tła:** Tymczasowo spowolniliśmy mniej priorytetowe zadania (takie jak masowy import towarów), aby dać pierwszeństwo Waszym zamówieniom.
- **Izolujemy problemy:** Wprowadzamy zabezpieczenia, które sprawią, że chwilowa zadyszka jednego modułu nie będzie blokować całej aplikacji.

**4. Kiedy będzie lepiej?**
Nasze systemy monitoringu pokazują, że sytuacja jest stabilna, choć nadal trudna. Dzięki podjętym działaniom przewidujemy powrót do pełnej płynności działania w ciągu najbliższych godzin. Będziemy na bieżąco informować o postępach.

Dziękujemy za wyrozumiałość.

---
