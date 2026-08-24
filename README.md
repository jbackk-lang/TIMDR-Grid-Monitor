# TIMDR-Grid-Monitor

Lokalne narzędzie badawczo-edukacyjne do monitoringu sieci energetycznej
metodą TIMDR na czterech kanałach: napięcie, częstotliwość, harmoniczne
(THD) i obciążenie. Wykrywa cztery typy zdarzeń: przeciążenia,
mikro-zaniki napięcia, anomalie harmoniczne i cykliczne zakłócenia.
Dodatkowo prognozuje kolejny odczyt i szacuje trend zużycia żywotności
kabla/linii pod danym profilem obciążenia.

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
   (zgodność ≥2 kanałów jednocześnie - licznik koincydencji, NIE fizyczny
   rezonans, patrz `ringdown.py` niżej).
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
   każdego kanału na kolejny krok (model Holta - wygładzanie z trendem,
   patrz niżej) + interpretacja zdarzeń prognozowanych (przewidywane
   przeciążenie/mikro-zanik/skok THD).
6. **Szacunek żywotności kabla/linii** (`cable_life.py`) - trend zużycia
   izolacji pod danym profilem obciążenia, dla dowolnego kabla (miedź
   lub aluminium, PVC/XLPE/EPR) - patrz "Model żywotności kabla" niżej.

## Detektory zdarzeń

- **Przeciążenia** - `load > overload_threshold × rated_load`
  (domyślnie 90%). Wymaga jawnego podania `rated_load` - nigdy cichego
  fallbacku na inną heurystykę.
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

## Rezonans częstotliwości (ringdown)

`grid_core.py::resonance()` to licznik koincydencji (ile kanałów zgłasza
anomalię jednocześnie) - nazwa pożyczona z fizyki, ale mechanizm inny.
`ringdown.py::ringdown_resonance()` liczy coś, co faktycznie odpowiada
fizycznemu rezonansowi: po zdarzeniu z `frequency_anomalies`
(`_detect_frequency_ringdown` w `grid_monitor.py`) sprawdza, czy powrót
częstotliwości do `f_nominal` (50Hz, podane JAWNIE, nie liczone ze
średniej okna - sieć ma realny punkt odniesienia) jest OSCYLACYJNY (sieć
"dzwoni" z powrotem do równowagi - dokładnie zjawisko znane w energetyce
jako tłumienie oscylacji mocy/częstotliwości po zaburzeniu, inter-area
oscillations) czy MONOTONICZNY (powrót bez dzwonienia - brak rezonansu).
Wynik trafia do `TimdrEnergyEvents.frequency_ringdown` (lista dictów:
`is_oscillatory`, `frequency_hz`, `damping_ratio`, `n_crossings`, itd.).

Port 1:1 matematyki z `jbackk-lang/universal-state-analyzer`
(`timdr_core/ringdown.py`) - tam pełna walidacja numeryczna na tłumionym
oscylatorze o znanej częstotliwości/stałej czasowej (patrz tamto README).
Metoda: histereza Schmitta NA WYKRYWANIU STANU (nie doklejona po fakcie do
już policzonych szczytów - pierwsza próba tak zrobiona dawała regresję,
patrz historia commitów tamtego repo) + interpolowane przejścia przez
poziom odniesienia + logarithmic decrement, częstotliwość liczona z
mediany (nie średniej) odstępów między przejściami.

**Regresja specyficzna dla tego repo, znaleziona przy porcie**: przy
realistycznej częstotliwości próbkowania sieci (fs=1000Hz) surowe
zero-crossing dawało dziesiątki-setki fałszywych "przejść" z powodu
drgania/chatter tuż przy prawdziwym przejściu przez zero (próbka szumu
tuż przy zerze kilkukrotnie zmienia znak, zanim sygnał wyraźnie odjedzie
na nową stronę) - dokładnie ten sam problem co w realnych komparatorach
analogowych bez histerezy. Zapisane w `test_ringdown.py`
(`test_underdamped_recovers_known_frequency_and_damping`, fs=1000Hz,
f0=0.5Hz, τ=1.5s - typowe rzędy wielkości dla oscylacji mocy w sieci) oraz
w `test_grid_monitor.py` (`test_frequency_ringdown_*`, integracja
end-to-end przez `analyze()`).

## Prognoza next-step

`forecast_core.py` przewiduje kolejny odczyt (napięcie/częstotliwość/
obciążenie/THD) na podstawie ostatniego okna historii i flaguje
przewidywane zdarzenia (`predicted_overload`/`predicted_micro_outage`/
`predicted_harmonic_spike`), analogicznie do detektorów w
`grid_monitor.py`, ale na wartości prognozowanej zamiast zmierzonej.

Model to **eksponencjalne wygładzanie z trendem (metoda Holta)** per
kanał, nie sieć neuronowa - zero dodatkowych zależności, w pełni
przetestowany, przewidywalny. Kontrakt (`predict_next()` zwraca 4
wartości) jest niezależny od implementacji, więc realny wytrenowany
model (LSTM czy inny, jeśli kiedyś pojawi się korpus rzeczywistych
danych do treningu) można podłączyć w przyszłości bez zmiany reszty
modułu - patrz "Ograniczenia".

## Model żywotności kabla

`cable_life.py` szacuje TREND zużycia żywotności izolacji kabla pod
danym profilem obciążenia, dla dowolnego materiału przewodnika
(miedź/aluminium) i typu izolacji (PVC/XLPE/EPR). Oparty o dwie
klasyczne zasady inżynierskie:

1. **Grzanie I²R** - przyrost temperatury przewodnika ponad otoczenie
   rośnie w przybliżeniu z kwadratem stosunku obciążenia do
   znamionowego (`(load/rated_load)²`). Uproszczenie pokrewne (nie
   identyczne) obwodowi cieplnemu z IEC 60287.

   **ZNALEZIONY I NAPRAWIONY BŁĄD:** pierwsza wersja liczyła przyrost
   przy obciążeniu znamionowym jako `rated_conductor_temp_c -
   ambient_temp_c`, czyli WZGLĘDEM TEGO SAMEGO `ambient_temp_c`, które
   zaraz potem było z powrotem dodawane do wyniku. Przy obciążeniu
   dokładnie znamionowym dawało to `temp = ambient + (rated - ambient) =
   rated` - wynik był CAŁKOWICIE NIEZALEŻNY od podniesienia/obniżenia
   temperatury otoczenia (dokładnie objaw, który zgłosił użytkownik: "nie
   wpływa na wynik"). Przy przeciążeniu było jeszcze gorzej - podniesienie
   ambient OBNIŻAŁO wynikową temperaturę przewodnika, fizycznie odwrotny
   kierunek. Naprawiono: przyrost przy obciążeniu znamionowym liczony jest
   teraz względem STAŁEJ `REFERENCE_AMBIENT_TEMP_C=25°C`, niezależnej od
   aktualnie wpisanego `ambient_temp_c` - dzięki temu podniesienie
   otoczenia o X stopni zawsze podnosi temperaturę przewodnika o dokładnie
   X stopni, dla dowolnego poziomu obciążenia. Regresja:
   `test_temp_rosnie_z_ambient_przy_dowolnym_obciazeniu` w
   `test_cable_life.py`.
2. **Reguła Montsingera/Arrheniusa** - żywotność izolacji polimerowej
   skraca się o połowę na każde `thermal_halving_deltaT_c` (domyślnie
   10°C) wzrostu temperatury ponad znamionową - ta sama logika co
   "reguła sześciu stopni" IEEE C57.91 dla transformatorów olejowych.

Domyślne stałe (temperatura znamionowa, projektowa żywotność) pochodzą
z typowych wartości wg konwencji IEC 60502 dla PVC/XLPE/EPR -
orientacyjne, nie zastępują karty katalogowej konkretnego kabla. **To
NIE jest certyfikowana kalkulacja wg IEC 60287/IEC 60216** - model nie
uwzględnia rzeczywistej rezystancji cieplnej otoczenia, wilgotności,
liczby cykli termicznych (tylko średnią temperaturę) ani stanu
mechanicznego kabla. Dla oceny realnej linii skonsultuj się z
uprawnionym elektroenergetykiem.

Współczynnik starzenia (`2^(ΔT/10)`, funkcja wykładnicza temperatury)
jest ograniczony do `MAX_AGING_FACTOR = 1000` (`cable_life.py`) - bez
tego capu pojedyncze próbki mocno powyżej `rated_load` potrafią
wywindować wynik do nieczytelnych wartości rzędu dziesiątek tysięcy,
bo uśrednianie funkcji wykładniczej nie spłaszcza ekstremów tak jak
średnia z wielkości liniowej. Powyżej tej wartości kabel w
rzeczywistości dawno przekroczyłby dopuszczalną temperaturę - większa
liczba nie niesie dodatkowej informacji. Przy dostrajaniu progów
pamiętaj o tej asymetrii między średnią temperaturą (liniowa) a
średnim współczynnikiem starzenia (wykładniczy).

## Uwagi techniczne (istotne przy dalszym rozwoju)

- **Status ALARM vs UWAGA** (`api.py`, `_summary()`): przeciążenia i
  mikro-zaniki zawsze dają ALARM. Anomalie THD/częstotliwości dają ALARM
  tylko gdy przekroczony jest twardy limit normy (`limit_normy`/
  `krytyczne`) - sama flaga adaptacyjna (MAD-z) daje UWAGA, nie ALARM,
  bo przy dużej liczbie próbek ma nieuniknioną statystyczną częstość
  "fałszywych" trafień. Przy dostrajaniu czułości pamiętaj o tym
  rozróżnieniu.
- **`rhythm()` (`grid_core.py`)**: znormalizowana autokorelacja per-lag
  ze ścisłymi wewnętrznymi maksimami lokalnymi + parametr `dip_frac`
  wymagający faktycznego spadku autokorelacji przed akceptacją
  kandydata na pik (odróżnia realną cykliczność od gładkich sygnałów
  z szumem). Ograniczenie: pik dokładnie na `max_lag` nigdy nie zostanie
  wykryty (brak prawego sąsiada) - `cyclic_max_lag` w scenariuszach demo
  ma z tego powodu margines nad oczekiwanym okresem.
- **Port 8070, celowo NIE 5060.** Port 5060 (i kilka innych: 5061, 6000,
  6666-6669, 6697) jest na liście "zakazanych portów" przeglądarek/
  `fetch()`.

## Struktura plików

```
TIMDR-Grid-Monitor/
├── grid_core.py          - prymitywy TIMDR (anomalies/defect/rhythm/resonance)
├── ringdown.py            - ringdown_resonance(): rezonans w sensie fizycznym (oscylacyjny powrót do f_nominal)
├── grid_monitor.py        - TimdrEnergyMonitor (interpretacja domenowa, 4 detektory + ringdown)
├── forecast_core.py       - prognoza next-step (Holt) + interpretacja zdarzeń prognozowanych
├── cable_life.py           - szacunek żywotności kabla (I²R + reguła Montsingera)
├── demo_generator.py      - 6 scenariuszy syntetycznych 230V/50Hz
├── csv_loader.py          - import CSV/Excel z walidacją schematu
├── device_client.py       - bufor + szkielet klientów Modbus/MQTT/REST
├── api.py                 - Flask API (port 8070) + serwowanie dashboardu
├── static/dashboard.html  - dashboard (ciemny motyw, Canvas 2D, bez CDN)
├── run.bat                - instalacja zależności + testy + start serwera
├── requirements.txt
└── test_*.py               - 114 testów pytest (w tym test_ringdown.py)
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

114/114 testów przechodzi (`grid_core`, `ringdown`, `grid_monitor`,
`forecast_core`, `cable_life`, `demo_generator` pośrednio przez
`grid_monitor`, `csv_loader`, `device_client`, `api`).

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
  z ostatniego okna, nie model wytrenowany na realnych, historycznych
  awariach - dobra do sygnalizowania "kierunku" najbliższego odczytu,
  nie do prognoz długoterminowych ani rzadkich zdarzeń, których nie było
  w analizowanym oknie.
- `ringdown_resonance()` (`ringdown.py`) zwalidowany wyłącznie na
  syntetycznym, czystym modelu tłumionego oscylatora (patrz
  universal-state-analyzer) - `noise_floor_factor=3.0` jest wartością
  ustaloną ręcznie, nieskalibrowaną na realnych danych sieciowych. Na
  sygnale z nakładającymi się częstotliwościami (kilka trybów oscylacji
  jednocześnie - realny scenariusz przy złożonych zakłóceniach
  wieloźródłowych) metoda może dać mylącą, uśrednioną częstotliwość -
  nietestowane.
- `cable_life.py` zakłada, że analizowane okno obciążenia reprezentuje
  TYPOWY profil pracy kabla - ekstrapolacja na cały pozostały okres
  życia jest tak dobra, jak reprezentatywność okna (60s danych demo ≠
  realny roczny profil obciążenia). Nie modeluje cykli termicznych ani
  stanu mechanicznego izolacji - tylko średnią temperaturę.
