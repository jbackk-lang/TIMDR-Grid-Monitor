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
├── demo_generator.py      - 6 scenariuszy syntetycznych 230V/50Hz
├── csv_loader.py          - import CSV/Excel z walidacją schematu
├── device_client.py       - bufor + szkielet klientów Modbus/MQTT/REST
├── api.py                 - Flask API (port 8070) + serwowanie dashboardu
├── static/dashboard.html  - dashboard (ciemny motyw, Canvas 2D, bez CDN)
├── run.bat                - instalacja zależności + testy + start serwera
├── requirements.txt
└── test_*.py               - 64 testy pytest
```

## Endpointy API

- `GET /` - dashboard
- `GET /api/health`
- `GET /api/scenarios` - lista dostępnych scenariuszy demo
- `GET /api/demo?scenario=...&rated_load=...` - analiza scenariusza syntetycznego
- `POST /api/csv` (multipart: `file`, `rated_load`, `load_unit`) - analiza wgranego pliku

## Testy

```
python -m pytest -q
```

64/64 testy przechodzą (`grid_core`, `grid_monitor` - w tym regresje
wszystkich 4 błędów ze szkicu źródłowego, `demo_generator` pośrednio
przez `grid_monitor`, `csv_loader`, `device_client`, `api`).

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
