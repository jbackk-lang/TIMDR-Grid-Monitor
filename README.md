# TIMDR-Grid-Monitor

Lokalne narzędzie badawczo-edukacyjne do monitoringu sieci energetycznej
metodą TIMDR na czterech kanałach: napięcie, częstotliwość, harmoniczne
(THD) i obciążenie. Wykrywa cztery typy zdarzeń: przeciążenia,
mikro-zaniki napięcia, anomalie harmoniczne i cykliczne zakłócenia.

> **Narzędzie badawczo-edukacyjne.** NIE zastępuje certyfikowanego
> analizatora jakości energii ani nie jest urządzeniem pomiarowym
> zgodnym z normą IEC 61000-4-30. Progi oparte o typowe wartości EN
> 50160 - przed użyciem produkcyjnym dostosuj do parametrów własnej
> instalacji.

## Uruchomienie

```
run.bat
```

Skrypt sam znajdzie Pythona (`python` albo `py`), doinstaluje zależności
(`flask numpy pandas openpyxl pytest`), uruchomi pełny zestaw testów i
odpali serwer + otworzy dashboard w przeglądarce pod
`http://127.0.0.1:8070`.

## Co robi monitor

1. **Silnik TIMDR** (`grid_core.py`) - te same cztery prymitywy co w
   innych projektach tej rodziny: `anomalies()` (MAD-z, odporny
   z-score), `defect()` (skok między kolejnymi próbkami względem
   rozstępu różnic + bezwzględna podłoga), `rhythm()` (okresowość przez
   znormalizowaną autokorelację + ścisłe lokalne maksima), `resonance()`
   (zgodność ≥2 kanałów jednocześnie).
2. **Interpretacja domenowa** (`grid_monitor.py`, `TimdrEnergyMonitor`)
   - cztery detektory zdarzeń per kanał, patrz niżej.
3. **Trzy źródła danych**: generator syntetyczny (`demo_generator.py`,
   6 scenariuszy 230V/50Hz zgodnych z EN 50160), import CSV/Excel
   (`csv_loader.py`, z walidacją schematu i jednostek), oraz szkielet
   pod realne urządzenie (`device_client.py`, Modbus/MQTT/REST -
   świadomie NIEzaimplementowane, patrz "Ograniczenia").
4. **Dashboard** - 4 wykresy kanałów, selektor scenariusza demo, upload
   CSV/Excel, log zdarzeń, plakietka statusu OK/UWAGA/ALARM.
5. **Prognoza next-step** (`forecast_core.py`) - przewidywana wartość
   każdego kanału na kolejny krok + interpretacja zdarzeń prognozowanych
   (przewidywane przeciążenie/mikro-zanik/skok THD).
6. **Szacunek żywotności kabla/linii** (`cable_life.py`) - trend zużycia
   izolacji pod danym profilem obciążenia, dla dowolnego kabla (miedź
   lub aluminium, PVC/XLPE/EPR) - patrz "Model żywotności kabla" niżej.

## Detektory zdarzeń

- **Przeciążenia** - `load > overload_threshold × rated_load`
  (domyślnie 90%). Wymaga jawnego podania `rated_load` - **nigdy cichego
  fallbacku** na inną heurystykę (patrz niżej, błąd 1 ze szkicu
  źródłowego).
- **Mikro-zaniki** - napięcie spada poniżej `micro_outage_drop ×
  v_nominal` (domyślnie 50%) na co najmniej `micro_outage_min_ms`
  (domyślnie 10ms, zgodnie z konwencją odświeżania RMS co pół okresu
  wg IEC 61000-4-30).
- **Anomalie harmoniczne** - podwójny próg: przekroczenie normy EN
  50160 (THD > 8%) LUB odchylenie adaptacyjne (MAD-z względem lokalnej
  mediany). Powód zdarzenia (`limit_normy` / `odchylenie_adaptacyjne` /
  `oba`) jest zwracany osobno - tylko przekroczenie normy eskaluje
  status dashboardu do ALARM (patrz "Uwagi techniczne").
- **Odchylenia częstotliwości** - pasmo normalne ±1% (49.5-50.5Hz) vs
  pasmo krytyczne ±4% (47-52Hz) od 50Hz nominalnych, zgodnie z EN 50160.
- **Cykliczne zakłócenia** - `rhythm()` na samym obciążeniu ORAZ osobno
  na syntetyzowanej serii binarnej zdarzeń (harmoniczne+częstotliwość),
  żeby wykryć regularność samych *zdarzeń*, nawet gdy obciążenie jest
  płaskie.

## Prognoza next-step

`forecast_core.py` przewiduje kolejny odczyt (napięcie/częstotliwość/
obciążenie/THD) na podstawie ostatniego okna historii i flaguje
przewidywane zdarzenia (`predicted_overload`/`predicted_micro_outage`/
`predicted_harmonic_spike`), analogicznie do detektorów w
`grid_monitor.py`, ale na wartości prognozowanej zamiast zmierzonej.

**Dlaczego NIE LSTM/PyTorch**, mimo że dostarczony szkic tak zakładał:
w środowisku budowy tego repo instalacja `torch` zakończyła się błędem
braku miejsca na dysku. Niezależnie od tej awarii - sieć z LOSOWO
zainicjalizowanymi wagami (żaden wytrenowany plik wag nie został
dostarczony) nie dałaby sensownej prognozy, tylko fikcyjne liczby
wyglądające jak działający model AI. Zamiast tego domyślny model to
**eksponencjalne wygładzanie z trendem (metoda Holta)** per kanał -
zero dodatkowych zależności, w pełni przetestowany, przewidywalny.
Kontrakt (`predict_next()` zwraca 4 wartości) jest niezależny od
implementacji - realny wytrenowany model (LSTM czy inny) można podłączyć
w przyszłości bez zmiany reszty modułu.

**Dwa błędy znalezione w dostarczonym szkicu** (`TimdrEnergyForecaster.
analyze_prediction`), oba tego samego rodzaju - porównanie prognozowanej
wartości DO SAMEJ SIEBIE zamiast do progu:

1. Przeciążenie: `load_next > overload_threshold * load_next` (x > 0.9x)
   jest PRAWDĄ dla każdej dodatniej wartości - gwarantowany fałszywy
   alarm na każdej pojedynczej prognozie, niezależnie od `rated_load`
   (które w ogóle nie było parametrem klasy). Naprawione: porównanie do
   `overload_threshold * rated_load`, `rated_load` wymagane w
   konstruktorze (jawny błąd, gdy brak - ta sama zasada co w
   `grid_monitor.TimdrEnergyMonitor`).
2. Skok THD: `harmonic_next > harmonic_spike_factor * harmonic_next`
   (x > 3x) jest prawdą TYLKO dla x < 0 - dla THD (zawsze ≥ 0) warunek
   nigdy się nie uruchamiał, czyli detektor był martwym kodem. Naprawione
   przez podwójny próg jak w `grid_monitor._detect_harmonic_anomalies` -
   stały limit normy EN 50160 (8%) ORAZ opcjonalny próg adaptacyjny
   (MAD-z) liczony z `recent_harmonics`, jeśli podane.

Test mikro-zaników w szkicu był poprawny (`voltage_next < drop *
v_nominal`) - nieporuszony.

## Model żywotności kabla

`cable_life.py` szacuje TREND zużycia żywotności izolacji kabla pod
danym profilem obciążenia, dla dowolnego materiału przewodnika
(miedź/aluminium) i typu izolacji (PVC/XLPE/EPR) - nie tylko dla
miedzi. Oparty o dwie klasyczne zasady inżynierskie:

1. **Grzanie I²R** - przyrost temperatury przewodnika ponad otoczenie
   rośnie w przybliżeniu z kwadratem stosunku obciążenia do
   znamionowego (`(load/rated_load)²`). Uproszczenie pokrewne (nie
   identyczne) obwodowi cieplnemu z IEC 60287.
2. **Reguła Montsingera/Arrheniusa** - żywotność izolacji polimerowej
   skraca się o połowę na każde `thermal_halving_deltaT_c` (domyślnie
   10°C) wzrostu temperatury ponad znamionową - ta sama logika co
   "reguła sześciu stopni" IEEE C57.91 dla transformatorów olejowych.

Domyślne stałe (temperatura znamionowa, projektowa żywotność) pochodzą
z typowych wartości wg konwencji IEC 60502 dla PVC/XLPE/EPR - orientacyjne,
nie zastępują karty katalogowej konkretnego kabla. **To NIE jest
certyfikowana kalkulacja wg IEC 60287/IEC 60216** - model nie uwzględnia
rzeczywistej rezystancji cieplnej otoczenia, wilgotności, liczby cykli
termicznych (tylko średnią temperaturę) ani stanu mechanicznego kabla.
Dla oceny realnej linii skonsultuj się z uprawnionym elektroenergetykiem.

**Błąd znaleziony i naprawiony: nieograniczony współczynnik starzenia
przy niedopasowanym `rated_load`.** Współczynnik starzenia to
`2^(ΔT/10)` - funkcja WYKŁADNICZA temperatury. Gdy profil obciążenia
zawiera choćby kilka próbek dużo powyżej `rated_load` (np. użytkownik
poda mniejszą moc znamionową niż realnie płynące waty w danych), model
I²R ekstrapoluje fizycznie absurdalną temperaturę (setki °C), a
uśrednienie funkcji wykładniczej po próbkach NIE spłaszcza takich
ekstremów tak, jak zrobiłaby to średnia z wielkości liniowej - stąd w
UI pozornie sprzeczny obraz: "średnia temperatura 81°C" obok
"współczynnik starzenia 89633×". Naprawione przez `MAX_AGING_FACTOR =
1000.0` (`cable_life.py`) - powyżej tej wartości kabel w rzeczywistości
dawno przekroczyłby dopuszczalną temperaturę (zadziałałoby
zabezpieczenie albo izolacja uległaby zniszczeniu), więc większa liczba
nie niesie dodatkowej informacji. Status `KRYTYCZNE` i tak się
utrzymuje dla chronicznego przeciążenia - zmienia się tylko czytelność
wyświetlanej liczby, nie klasyfikacja.

## Uwagi techniczne (istotne przy dalszym rozwoju)

- **Cztery błędy znalezione w dostarczonym szkicu kodu** (przed
  zbudowaniem tego repo szkic został zweryfikowany empirycznie, nie
  zaufany bezpośrednio): (1) przeciążenia liczone względem `max()` z
  bieżącego okna zamiast `rated_load` - flagowały ~21% zdrowego profilu
  jako "przeciążenie"; naprawione przez wymaganie jawnego
  `rated_load` i jawny błąd, gdy go brak. (2) częstotliwość w ogóle nie
  była analizowana mimo istnienia w sygnaturze - dodany pełny detektor
  z dwoma pasmami EN 50160. (3) próg THD wyłącznie adaptacyjny (3×
  mediana) przeoczał realne przekroczenie normy 8% na sieci z
  przewlekle podwyższonym tłem (mediana 6% → próg 18% > wstrzyknięte
  9.5%) - naprawione przez podwójny próg (norma OR adaptacyjny).
  (4) detekcja cykliczności na surowym niedetrendowanym sygnale fałszywie
  łapała trend liniowy jako "okresowość" - naprawione przez detrend +
  ścisłe lokalne maksima wewnętrzne.
- **`rhythm()` (własna implementacja w `grid_core.py`)**: pierwsza wersja
  powtórzyła dwa błędy znane z poprzednich projektów TIMDR w tym
  zestawie - nieznormalizowana korelacja z granicznym lagiem trywialnie
  wygrywającym lokalne maksimum, oraz fałszywa "okresowość" na gładkich
  sygnałach (długi okres dobowy + szum daje płaskie wysokie
  autokorelacje przy krótkich lagach, gdzie szum numeryczny tworzy
  pozorne maksima lokalne). Naprawione przez port sprawdzonego algorytmu
  ze `starszego` repo (znormalizowana korelacja per-lag, ścisłe
  wewnętrzne maksima) plus nowy parametr `dip_frac` wymagający
  faktycznego spadku autokorelacji poniżej progu przed akceptacją
  kandydata na pik. Ograniczenie odziedziczone (nie regresja): pik
  dokładnie na `max_lag` nigdy nie zostanie wykryty (brak prawego
  sąsiada) - `cyclic_max_lag` w scenariuszach demo ma z tego powodu
  margines nad oczekiwanym okresem.
- **Status ALARM vs UWAGA** (`api.py`, `_summary()`): przeciążenia i
  mikro-zaniki zawsze dają ALARM. Anomalie THD/częstotliwości dają ALARM
  tylko gdy przekroczony jest twardy limit normy (`limit_normy`/
  `krytyczne`) - sama flaga adaptacyjna (MAD-z) daje UWAGA, nie ALARM,
  bo przy dużej liczbie próbek ma nieuniknioną statystyczną częstość
  "fałszywych" trafień (np. ~12 na 6000 próbek scenariusza "typowa
  praca" przy factor=3.0 - to oczekiwane zachowanie progu adaptacyjnego,
  nie błąd). Przy dostrajaniu czułości pamiętaj o tym rozróżnieniu.
- **Port 8070, celowo NIE 5060.** Ta sama pułapka co w innych projektach
  tego zestawu - port 5060 (i kilka innych: 5061, 6000, 6666-6669, 6697)
  jest na liście "zakazanych portów" przeglądarek/`fetch()`.

## Struktura plików

```
TIMDR-Grid-Monitor/
├── grid_core.py          - prymitywy TIMDR (anomalies/defect/rhythm/resonance)
├── grid_monitor.py        - TimdrEnergyMonitor (interpretacja domenowa, 4 detektory)
├── forecast_core.py       - prognoza next-step (Holt) + interpretacja zdarzeń prognozowanych
├── cable_life.py           - szacunek żywotności kabla (I²R + reguła Montsingera)
├── demo_generator.py      - 6 scenariuszy syntetycznych 230V/50Hz
├── csv_loader.py          - import CSV/Excel z walidacją schematu
├── device_client.py       - bufor + szkielet klientów Modbus/MQTT/REST
├── api.py                 - Flask API (port 8070) + serwowanie dashboardu
├── static/dashboard.html  - dashboard (ciemny motyw, Canvas 2D, bez CDN)
├── run.bat                - instalacja zależności + testy + start serwera
├── requirements.txt
└── test_*.py               - 106 testów pytest
```

## Endpointy API

- `GET /` - dashboard
- `GET /api/health`
- `GET /api/scenarios` - lista dostępnych scenariuszy demo
- `GET /api/demo?scenario=...&rated_load=...&<cable_params>` - analiza scenariusza syntetycznego
- `POST /api/csv` (multipart: `file`, `rated_load`, `load_unit`, `<cable_params>`) - analiza wgranego pliku

`<cable_params>` (opcjonalne, wpływają na `cable_life` w odpowiedzi):
`insulation_type` (pvc/xlpe/epr), `conductor_material` (copper/aluminum),
`ambient_temp_c`, `years_in_service`, `thermal_halving_deltaT_c`,
`design_life_years`.

Odpowiedź obu endpointów analizy zawiera, oprócz `signals`/`events`/
`summary`: `forecast` (`prediction` + `events`) i `cable_life`.

## Testy

```
python -m pytest -q
```

106/106 testów przechodzi (`grid_core`, `grid_monitor` - w tym regresje
wszystkich 4 błędów ze szkicu źródłowego, `forecast_core` - w tym
regresje 2 błędów ze szkicu predyktora, `cable_life`, `demo_generator`
pośrednio przez `grid_monitor`, `csv_loader`, `device_client`, `api`).

## Ograniczenia

- `ModbusGridClient`/`MqttGridClient`/`RestGridClient` w
  `device_client.py` są świadomie NIEzaimplementowane - rzucają
  `NotImplementedError` z konkretną instrukcją, co dopisać. Nie ma tu
  żadnej symulacji/udawania realnych danych pod płaszczem "działającego"
  klienta - mapowanie rejestrów Modbus, format payloadu MQTT i schemat
  REST różnią się między urządzeniami i nie da się ich sensownie
  zgadnąć bez dokumentacji konkretnego licznika/PLC.
- Wykrywanie mikro-zaników zakłada próbkowanie wystarczająco gęste, by
  uchwycić zdarzenia rzędu 10ms (IEC 61000-4-30) - dane z CSV o rzadkim
  próbkowaniu (np. co 1 min) nigdy nie wykryją mikro-zaników, nawet jeśli
  faktycznie wystąpiły.
- Import CSV/Excel wymaga kolumny czasu (do wyliczenia `sample_rate_hz`)
  ALBO jawnego podania `sample_rate_hz` - bez żadnego z nich rzuca błąd,
  nie zgaduje częstotliwości próbkowania.
- Progi EN 50160 użyte tu (230V±10%, 50Hz±1%/±4%, THD≤8%) to typowe
  wartości normy dla sieci niskiego napięcia - realne umowy przyłączeniowe
  mogą mieć własne, bardziej rygorystyczne limity.
- Prognoza (`forecast_core.py`) to statystyczna ekstrapolacja trendu
  (metoda Holta) z ostatniego okna, NIE model AI wytrenowany na
  realnych, historycznych awariach - dobra do sygnalizowania "kierunku"
  najbliższego odczytu, nie do prognoz długoterminowych ani rzadkich
  zdarzeń, których nie było w analizowanym oknie.
- `cable_life.py` zakłada, że analizowane okno obciążenia reprezentuje
  TYPOWY profil pracy kabla - ekstrapolacja `mean_aging_factor` na cały
  pozostały okres życia jest tak dobra, jak reprezentatywność okna
  (60s danych demo ≠ realny roczny profil obciążenia). Nie modeluje też
  cykli termicznych ani stanu mechanicznego izolacji - tylko średnią
  temperaturę.
