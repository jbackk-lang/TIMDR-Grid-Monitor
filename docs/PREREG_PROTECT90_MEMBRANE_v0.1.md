# Pre-rejestracja: protect90_membrane_bridge — widmo macierzy korelacji między jednoczesnymi fazami prądu (i napięcia) sieci PROTECT-90 jako obiekt G, na nowej domenie (baseline vs event zwarciowy)

> Status: PRE-REJESTRACJA, zamrożona PRZED policzeniem jakiejkolwiek
> metryki na realnych epizodach PROTECT-90 (kontrole syntetyczne z
> `spectral_family.py` dla N=2,3 są już znane z wcześniejszej sesji na
> łożyskach CWRU — reużyte tu jako gotowa walidacja mechanizmu, patrz
> §5 — ale N=6 zostaje policzone i zacytowane tutaj PRZED dotknięciem
> danych PROTECT-90). Data: 2026-09-22. Nowa domena (sieć
> elektroenergetyczna, zwarcia), NIE kontynuacja `chrono_membrane_bridge`
> na łożyskach — kod matematyczny identyczny (reużyty 1:1 z
> `timdr_geometry/spectral_family.py`), ale świeża pre-rejestracja z
> własną hipotezą kierunkową, bo domena fizyczna jest inna.

## 0. Skąd to się bierze

`chrono_membrane_bridge` (`docs/geometry/PREREG_CHRONO_MEMBRANE_BEARING_v0.1.md`
w `GIA-TIMDR`) ustalił, że widmo macierzy korelacji Pearsona między
jednocześnie próbkowanymi kanałami (prawdziwa rodzina Chronoprocesu
`{γ_s}_{s∈I}`, `s`=kanał, `t`=próbka) jest mechanizmem, który: (a) na
kontroli syntetycznej poprawnie odróżnia rosnące/stałe/brak sprzężenie
międzykanałowe bez fałszywych alarmów (6/6 komórek PASSED na łożyskach,
N∈{2,3}), (b) na realnych danych łożysk CWRU dał SILNY i STABILNY
(6/6 komórek, znak identyczny na oknach 128/256/512) efekt na
`spectral_concentration`, ale w kierunku PRZECIWNYM do pierwotnej
hipotezy (`fault < normal`, nie `fault > normal`) — formalnie
NOT_SUPPORTED mimo siły efektu, zgłoszone w pełni w
`RESULT_CHRONO_MEMBRANE_BEARING_v0.1.md`. Kod matematyczny został od
tego czasu wydzielony 1:1 do domenowo-agnostycznego modułu
`TIMDR-Geometry-Formalism/timdr_geometry/spectral_family.py` — ten sam
plik (identyczna treść) istnieje też skopiowany do samodzielnego repo
`TIMDR-Geometry-Formalism` obok `GIA-TIMDR`.

Ta sesja stosuje TEN SAM, już zwalidowany kod (`channel_correlation_matrix`,
`spectrum_from_correlation`, `spectral_concentration`, `participation_ratio`,
`membrane_window_metrics`, `run_membrane_controls`, `run_synthetic_controls`)
do NOWEJ domeny — PROTECT-90 (symulacja EMT sieci wysokiego napięcia,
zwarcia trójfazowe), NIE do łożysk. Kod NIE jest przepisywany ani
modyfikowany — importowany bezpośrednio z `spectral_family.py`.
Kluczowa lekcja metodologiczna z serii `chrono_membrane_bridge` (§0
tamtego PREREG-u) — G musi czytać PRAWDZIWĄ rodzinę wielu jednoczesnych
kanałów, nie jeden sygnał sztucznie zembedowany — jest tu spełniona
wprost: `s` indeksuje FAZĘ (L1/L2/L3 prądu, opcjonalnie też napięcia),
`t` indeksuje próbkę czasu wewnątrz epizodu 1-sekundowego, próbkowane
JEDNOCZEŚNIE w tej samej lokalizacji sieci.

## 1. Audyt nazw PRZED użyciem

Grep na `TIMDR-Grid-Monitor`, `GIA-TIMDR`, `TIMDR-Geometry-Formalism`:
`protect90_membrane`, `PROTECT90_MEMBRANE` — **zero kolizji**. Nowa
nazwa `protect90_membrane_bridge` (plik
`real_protect90_membrane_bridge.py`) odróżniona jawnie od
`chrono_membrane_bridge` (domena łożysk CWRU, `core/` w `GIA-TIMDR`) —
ten sam matematyczny mechanizm (widmo macierzy korelacji), inna domena
danych, inny plik, inna pre-rejestracja. Nazwy metryk
(`spectral_concentration`, `participation_ratio`, `membrane_spectral_ratio`)
są REUŻYWANE bez zmian z `spectral_family.py` — to ten sam obiekt
matematyczny, więc reużycie nazwy jest tu poprawne (w odróżnieniu od
`chrono_cone_ratio`/`funnel_ratio` w historii repo, gdzie CELOWO
unikano tej samej nazwy dla różnych konstrukcji).

## 2. Krok 1: dane — co jest dostępne (audytowane PRZED wyborem kanałów)

Źródło: `TIMDR-Grid-Monitor/data/protect90/frozen/`.

- `B4_GRID_PROTECT90_FROZEN_MANIFEST.json`: dataset PROTECT-90 v1.0.0
  (DOI `10.5281/zenodo.21109169`), symulacja EMT sieci WN (NIE dane
  polowe — `source_type`), `sampling_rate_hz=6400`,
  `episode_duration_seconds=1.0`, `n_selected_episodes=512`.
  `analysis_rule` (już ustalona konwencja tego repo, zastosowana tu
  wprost, nie wymyślona na nowo): *"compare pre-event baseline with
  event-aligned windows; do not tune selection after results"*.
- `B4_GRID_PROTECT90_FROZEN_EPISODE_IDS.json`: 512 zamrożonych ID
  epizodów (`selection_rule`: "128 episode IDs per sc_type, SHA-256
  ordered before waveform access", 4 typy zwarcia × 128 = 512, wybór
  NIEZALEŻNY od tej sesji, dokonany wcześniej). Te ID są UŻYWANE bez
  zmian — żadnej nowej selekcji.
- `preprocessed_data/<id>_sample_hv_double_line_90kv.pkl`: DataFrame
  6400×49, `time_s` + 48 kanałów = 8 lokalizacji × (prąd 3-fazowy +
  napięcie 3-fazowy). `time_s[i] = (i+1)/6400` (potwierdzone
  bezpośrednio: `time_s[0]=0.000156`, `time_s[6399]=1.0`).
- `raw/hv_double_line_90kv_labels.csv`: `t_evnt_start`, `t_evnt_end`
  (sekundy wewnątrz epizodu 1 s), `sc_type` (typ zwarcia, 0-3),
  `sc_location`, `phase_select`, `fault_target`.

### 2.1 Wybór lokalizacji — reguła NIEZALEŻNA od wyniku

Reguła zapisana PRZED spojrzeniem na jakikolwiek wynik: **pierwsza
alfabetycznie nazwa lokalizacji wśród 8 dostępnych**. Osiem lokalizacji
obecnych w kolumnach pliku (potwierdzone bezpośrednio z `4_sample_...pkl`):
`Bus_1_Line_01_02A`, `Bus_1_Line_01_02B`, `Bus_2_Line_01_02A`,
`Bus_2_Line_01_02B`, `Bus_2_Line_02_03A`, `Bus_2_Line_02_03B`,
`Bus_3_Line_02_03A`, `Bus_3_Line_02_03B` — alfabetycznie pierwsza:
**`Bus_1_Line_01_02A`**. Ta sama lokalizacja używana w OBU testach
(primarnym N=3 i sekundarnym N=6) — reguła wyboru nie zależy od tego,
która lokalizacja "wygląda dobrze" na żadnej metryce.

- **Test PRIMARNY (N=3)**: `Bus_1_Line_01_02A_cur_{L1,L2,L3}_A` — prąd
  trójfazowy tej lokalizacji, trzy JEDNOCZEŚNIE próbkowane kanały,
  prawdziwa rodzina `{γ_s}_{s∈I}`, `I`={L1,L2,L3}.
- **Test SEKUNDARNY, bogatszy (N=6)**: ta sama lokalizacja, prąd +
  napięcie trójfazowe razem (`cur_L1,cur_L2,cur_L3,vol_L1,vol_L2,vol_L3`).
  `spectral_family.py` przyjmuje `Sequence[np.ndarray]` o dowolnej
  długości — funkcje (`channel_correlation_matrix`, `spectrum_from_correlation`,
  `spectral_concentration`, `participation_ratio`, `membrane_window_metrics`)
  NIE wymagają żadnej zmiany dla N=6, sprawdzone bezpośrednio (§5,
  kontrola syntetyczna N=6 przechodzi bez modyfikacji kodu).

## 3. Krok 2: hipoteza i okna (zamrożone PRZED uruchomieniem na realnych epizodach)

### 3.1 Definicja okien — baseline vs event, zgodnie z `analysis_rule`

Dla każdego epizodu: `event_start_idx` = pierwszy indeks próbki z
`time_s >= t_evnt_start` (wyszukiwanie binarne na `time_s`, nie
zaokrąglenie ręczne). **Baseline** = `window_size` próbek
BEZPOŚREDNIO PRZED `event_start_idx` (`[event_start_idx - window_size,
event_start_idx)`). **Event** = `window_size` próbek BEZPOŚREDNIO PO
`event_start_idx` (`[event_start_idx, event_start_idx + window_size)`,
zgodnie z regułą "event-aligned windows"). Ten sam `window_size` po
obu stronach — porównanie PAROWANE (ten sam epizod, dwa okna), nie
dwie niezależne grupy.

**Filtr długości** (odrzucenie PRZED analizą, nie po wyniku): epizod
odrzucony, jeśli `event_start_idx < window_size` (za mało próbek na
baseline) LUB `event_start_idx + window_size > 6400` (za mało próbek
na event). Sprawdzone na WSZYSTKICH 512 zamrożonych epizodach PRZED
napisaniem reszty tego dokumentu: `t_evnt_start` w tym zbiorze mieści
się w `[0.2002, 0.4997]` s (`min=0.2002`, `max=0.4997`), czyli
`event_start_idx` w przybliżeniu `[1282, 3198]` — dla obu rozmiarów
okna użytych w tej sesji (§3.2, `window_size∈{128,256}`, próg
`256/6400=0.04` s) WSZYSTKIE 512 epizodów mają wystarczający zapas z
obu stron. **Oczekiwana liczba odrzuconych: 0/512** — ale filtr
zostaje w kodzie i jest jawnie raportowany per rozmiar okna, na
wypadek gdyby to założenie okazało się błędne przy faktycznym
uruchomieniu.

### 3.2 Rozmiary okna

`WINDOW_SIZES = (128, 256)` próbek przy `fs=6400` Hz → `20 ms` i
`40 ms`, czyli ok. 1 i 2 okresy `50 Hz` — fizycznie sensowne dla
zjawisk zwarciowych (krótsze niż siatka łożyskowa `{128,256,512}`,
bo epizod PROTECT-90 ma tylko 6400 próbek/1 s, a zdarzenie zwarciowe
jest zlokalizowane w czasie, nie chcemy okna event dłuższego niż sam
epizod pozwala). `512` (80 ms, 4 okresy) pominięty dla testu
primarnego, żeby nie rozmywać okna event na obszar znacznie poza
najbardziej dynamiczną fazą zwarcia — ale sprawdzony jako TRZECI,
dodatkowy rozmiar w kontroli syntetycznej N=6 (§5, taka sama siatka
jak reszta rodziny `spectral_family`, `{128,256,512}`) dla spójności
z resztą ekosystemu.

### 3.3 Wybór testu statystycznego — Wilcoxon sparowany (uzasadnienie z góry)

Baseline i event pochodzą z TEGO SAMEGO epizodu (naturalne parowanie,
nie dwie niezależne grupy jak normal/fault w domenie łożyskowej) —
**test Wilcoxona ze znakiem rang (sparowany, dwustronny)** jest tu
metodologicznie poprawniejszy niż Mann-Whitney U, bo wykorzystuje
strukturę parowania (każdy epizod jako własna kontrola) zamiast
traktować 512 wartości baseline i 512 wartości event jako niezależne
próby. Implementacja: `scipy.stats.wilcoxon(method="approx")` (zwraca
`zstatistic`), rozmiar efektu = macierzowo-parowany rank-biserial
`r = z/sqrt(n)` (konwencja dla testu Wilcoxona, analogiczna do
`rank_biserial_effect_size` z `timdr_formalism/pipeline.py` dla
Manna-Whitneya, ale dla par, nie dwóch niezależnych grup) — progi
efektu (`effect_size_label`) reużyte bez zmian z `pipeline.py`.
`SUPPORTED` = `p<0.05` i `|r|≥0.3`, tak jak konwencja repo.

### 3.4 Hipoteza kierunkowa — PRZED danymi, z jawnym zastrzeżeniem niepewności

**Robocze przewidywanie**: zwarcie (silne, nagłe zaburzenie WSPÓLNE dla
wszystkich trzech faz na dotkniętej linii — prąd zwarciowy widoczny
symultanicznie na L1/L2/L3, niezależnie od typu zwarcia) powinno
ZWIĘKSZYĆ `spectral_concentration` w oknie event względem baseline
(kanały fazowe stają się silniej wzajemnie skorelowane pod wpływem
wspólnego, dominującego zaburzenia) — analogicznie do pierwotnej
(odrzuconej) hipotezy z domeny łożyskowej, gdzie spodziewano się, że
uszkodzenie zwiększy koncentrację.

**Jawne zastrzeżenie, zapisane PRZED uruchomieniem**: dokładnie ta sama
klasa przewidywania ("silne wspólne zaburzenie → wyższa koncentracja")
została w domenie łożyskowej OBALONA — wynik poszedł w kierunku
PRZECIWNYM (`fault < normal`), z fizycznie wiarygodnym uzasadnieniem
post-hoc (§3.1 `RESULT_CHRONO_MEMBRANE_BEARING_v0.1.md`: lokalny
defekt może ROZPRZĘGAĆ kanały przez różne ścieżki transmisji, zamiast
je sprzęgać). Sieć elektroenergetyczna i łożysko to fizycznie różne
systemy (zwarcie trójfazowe ma inny mechanizm sprzężenia międzyfazowego
niż propagacja wibracji mechanicznej przez obudowę łożyska), więc
kierunek NIE jest tu ekstrapolowany wprost z wyniku łożyskowego — ale
doświadczenie z tamtej sesji nakazuje jawną ostrożność: **kierunek tego
przewidywania jest NIEPEWNY z góry, wynik w KAŻDĄ stronę (w tym
przeciwną do przewidywania) jest w pełni poprawną, akceptowaną
odpowiedzią tej sesji**, zgodnie z klasyfikacją §7.

### 3.5 Test stabilności znaku

Rank-biserial `r` (na `spectral_concentration`, test primarny N=3)
porównany między `WINDOW_SIZES=(128,256)` dla łączonej próby 512
epizodów. **ZNAK STABILNY** = ten sam znak na obu rozmiarach okna.
**ZNAK NIESTABILNY** = różny znak — to jest KLUCZOWE kryterium całej
serii `chrono_*`/`membrane_*` tego ekosystemu (v0.1/v0.2 łożyskowe:
niestabilny; `chrono_membrane_bridge` łożyskowy: stabilny), sprawdzane
tu z tą samą dyscypliną, bez zakładania wyniku z góry.

## 4. Krok 3: kontrole syntetyczne (reużyte, nie nowe)

`run_synthetic_controls()` z `spectral_family.py` (bez zmian) już
przeszedł dla `N∈{2,3}` × `window∈{128,256,512}` — 6/6 komórek PASSED,
zreplikowane bezpośrednio w tej sesji PRZED napisaniem tego dokumentu
(identyczne liczby jak w oryginalnej walidacji łożyskowej, bo kod i
generatory syntetyczne są domenowo-agnostyczne — nie zależą od
PROTECT-90 ani CWRU):

| N | window | pozytywna (a vs b) | negatywna (b vs c) | PASSED |
|---|---|---|---|---|
| 2 | 128 | p=5.49e-11, r=0.987 | p=0.819, r=0.036 | TAK |
| 2 | 256 | p=3.02e-11, r=1.000 | p=0.284, r=0.162 | TAK |
| 2 | 512 | p=1.03e-06, r=0.736 | p=0.277, r=-0.164 | TAK |
| 3 | 128 | p=3.02e-11, r=1.000 | p=0.333, r=-0.147 | TAK |
| 3 | 256 | p=3.02e-11, r=1.000 | p=0.877, r=0.024 | TAK |
| 3 | 512 | p=6.52e-09, r=0.873 | p=0.923, r=0.016 | TAK |

**N=6** (test sekundarny tej sesji, cur+vol trójfazowy) — policzone
DODATKOWO w tej sesji, PRZED dotknięciem danych PROTECT-90, wywołaniem
`run_membrane_controls(n_channels=6, ...)` z TYMI SAMYMI parametrami
(`n_windows=30, seed=0, alpha=0.05`) co reszta siatki, żadnego nowego
kodu:

| N | window | pozytywna (a vs b) | negatywna (b vs c) | median conc.(b) | median conc.(c) | 1/N | PASSED |
|---|---|---|---|---|---|---|---|
| 6 | 128 | p=3.02e-11, r=1.000 | p=0.420, r=0.122 | 0.2142 | 0.6112 | 0.1667 | TAK |
| 6 | 256 | p=3.02e-11, r=1.000 | p=0.130, r=0.229 | 0.2034 | 0.6072 | 0.1667 | TAK |
| 6 | 512 | p=5.57e-10, r=0.933 | p=0.473, r=0.109 | 0.1901 | 0.6044 | 0.1667 | TAK |

**9/9 komórek PASSED (N∈{2,3,6} × window∈{128,256,512})** — mechanizm
poprawnie wykrywa rosnące sprzężenie (pozytywna), nie daje fałszywego
alarmu na stałym poziomie korelacji (negatywna), i skaluje się do N=6
bez żadnej zmiany kodu. Przejście do realnych danych PROTECT-90
uzasadnione dla obu testów (primarnego N=3 i sekundarnego N=6).

## 5. Krok 4: klasyfikacja końcowa (ustalona z góry)

**SUPPORTED** = `p<0.05`, `|r|≥0.3` (Wilcoxon sparowany, §3.3), ZNAK
STABILNY między `window_size∈{128,256}` (§3.5), kierunek zgodny z
przewidywaniem §3.4 (event > baseline na `spectral_concentration`).
**NOT_SUPPORTED** = `p≥0.05` przy wystarczającej liczbie obserwacji
(≥10 par, konwencja repo) LUB kierunek przeciwny do przewidywania
(nawet przy silnym, istotnym efekcie — zgodnie z precedensem
`chrono_membrane_bridge` łożyskowego, §3.4 wyżej). **NIESTABILNY** =
formalnie istotny na obu oknach, ale ZNAK się różni między nimi —
traktowane jako BRAK potwierdzenia stabilnej struktury, nie jako
częściowy sukces. **INCONCLUSIVE** = <10 ważnych par w którymś oknie
(np. po odrzuceniu epizodów z za krótkim baseline/event, §3.1).
`participation_ratio` i `membrane_spectral_ratio` raportowane obok
jako diagnostyka (jak w PREREG łożyskowym), NIE zmieniają klasyfikacji.

Klasyfikacja robiona OSOBNO dla testu primarnego (N=3, prąd) i
sekundarnego (N=6, prąd+napięcie) — jawne porównanie N=3 vs N=6 w
raporcie końcowym, bez ważenia jednego wynikiem drugiego.

## 6. Status

Zamrożone. Kolejność wykonania: (1) ten dokument (już zapisany); (2)
`real_protect90_membrane_bridge.py` w `TIMDR-Grid-Monitor/` — wczytanie
512 zamrożonych epizodów, obliczenie baseline/event dla lokalizacji
`Bus_1_Line_01_02A`, N=3 i N=6, oba rozmiary okna, test Wilcoxona
sparowany; (3) `docs/RESULT_PROTECT90_MEMBRANE_v0.1.md` z pełnymi
liczbami, wynikiem pozytywnym/negatywnym/częściowym — którykolwiek
wyjdzie — bez retuningu po fakcie; (4) test smoke + commit lokalny
(bez push).
