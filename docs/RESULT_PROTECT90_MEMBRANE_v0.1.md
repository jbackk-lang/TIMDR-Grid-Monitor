# Wynik: protect90_membrane_bridge — widmo macierzy korelacji między fazami prądu (i napięcia) sieci PROTECT-90, baseline vs event zwarciowy. Kontrole syntetyczne PRZESZŁY (9/9 komórek, N∈{2,3,6}). Na 512 zamrożonych epizodach: SILNY, STABILNY efekt na `spectral_concentration`, w kierunku ZGODNYM z przewidywaniem a priori — formalnie SUPPORTED dla N=3 (primarny) i N=6 (sekundarny) (2026-09-22)

> Kontynuacja `PREREG_PROTECT90_MEMBRANE_v0.1.md` (zamrożonej PRZED
> uruchomieniem czegokolwiek na danych PROTECT-90). Kod matematyczny
> reużyty 1:1, bez zmian, z `TIMDR-Geometry-Formalism/timdr_geometry/spectral_family.py`
> (już zwalidowany na domenie łożyskowej). Kod tej sesji:
> `real_protect90_membrane_bridge.py`. Nic w żadnym z plików nie zostało
> zmienione PO zobaczeniu wyników poniżej — jedyna poprawka wprowadzona
> w trakcie sesji (§2.1) dotyczyła błędu w SPOSOBIE ODCZYTU znaku ze
> zwracanej wartości biblioteki `scipy`, wykrytego i naprawionego PRZED
> spojrzeniem na skorygowane liczby, udokumentowanego w pełni poniżej.

## 0. Rekapitulacja konstrukcji

`s` indeksuje FAZĘ (L1/L2/L3 prądu, primarny test N=3; + L1/L2/L3
napięcia, sekundarny test N=6), `t` indeksuje próbkę czasu wewnątrz
epizodu 1-sekundowego (`fs=6400` Hz), lokalizacja `Bus_1_Line_01_02A`
(pierwsza alfabetycznie z 8 dostępnych, reguła ustalona PRZED
wynikiem, PREREG §2.1). Dla każdego z 512 zamrożonych epizodów:
macierz korelacji Pearsona `C` między `N` jednoczesnymi kanałami na
oknie BASELINE (`window_size` próbek bezpośrednio PRZED
`t_evnt_start`) i oknie EVENT (`window_size` próbek bezpośrednio PO),
widmo `λ_1≥...≥λ_N≥0`, `spectral_concentration=λ_1/Σλ`,
`participation_ratio=(Σλ)²/Σλ²`, `membrane_spectral_ratio` (brzeg/brzeg
wewnątrz okna, `edge_fraction=0.2`) — wszystko z `membrane_window_metrics()`,
BEZ ŻADNEJ zmiany kodu względem domeny łożyskowej.

## 1. Krok 1: audyt danych — potwierdzone bezpośrednio

- 512/512 zamrożonych epizodów wczytanych, **0 brakujących plików**.
- Filtr długości okna (PREREG §3.1): `event_start_idx` w zbiorze
  512 epizodów mieści się w `[1282, 3198]` (`t_evnt_start∈[0.2002,
  0.4997]` s) — dla `WINDOW_SIZES=(128,256)` **0/512 epizodów
  odrzuconych** na którymkolwiek rozmiarze okna, dokładnie zgodnie z
  przewidywaniem PREREG §3.1. `state["rejected"]={}`.
- Rozkład 512 zamrożonych epizodów: dokładnie 128 na każdy z 4
  `sc_type` (0,1,2,3) — potwierdzone niezależnie z `labels.csv` po
  złączeniu z `B4_GRID_PROTECT90_FROZEN_EPISODE_IDS.json`.

## 2. Kontrole syntetyczne — PRZESZŁY na WSZYSTKICH 9 komórkach (N∈{2,3,6} × window∈{128,256,512})

Zreplikowane w tej sesji z `spectral_family.py` bez zmian kodu (N=2,3:
identyczne liczby jak w oryginalnej walidacji łożyskowej — kod jest
domenowo-agnostyczny, jak oczekiwano; N=6: policzone dodatkowo w tej
sesji, PRZED dotknięciem danych PROTECT-90, jednym wywołaniem
`run_membrane_controls(n_channels=6, ...)`):

| N | window | pozytywna (a vs b) | negatywna (b vs c) | PASSED |
|---|---|---|---|---|
| 2 | 128/256/512 | p=5.49e-11 / 3.02e-11 / 1.03e-06, r duży | p=0.82 / 0.28 / 0.28 | TAK/TAK/TAK |
| 3 | 128/256/512 | p=3.02e-11 / 3.02e-11 / 6.52e-09, r duży | p=0.33 / 0.88 / 0.92 | TAK/TAK/TAK |
| 6 | 128/256/512 | p=3.02e-11 / 3.02e-11 / 5.57e-10, r duży | p=0.42 / 0.13 / 0.47 | TAK/TAK/TAK |

Pełne liczby: `docs/PREREG_PROTECT90_MEMBRANE_v0.1.md` §4. Mechanizm
poprawnie skaluje się do N=6 bez żadnej zmiany kodu (funkcje przyjmują
`Sequence[np.ndarray]` dowolnej długości, jak przewidziano). Przejście
do danych realnych uzasadnione dla obu testów.

## 2.1 Błąd wykryty i naprawiony w trakcie sesji — znak `zstatistic` z `scipy.stats.wilcoxon`

Pierwsze uruchomienie testu głównego dało dla WSZYSTKICH komórek
`spectral_concentration` znak `r<0` (event < baseline), co byłoby
wynikiem SPRZECZNYM z przewidywaniem. Przed zapisaniem tego jako wynik
sprawdzono mechanizm: `scipy.stats.wilcoxon(..., alternative="two-sided",
method="approx")` zwraca `zstatistic` PRZEPUSZCZONY przez `z=-abs(z)`
wewnątrz `scipy/stats/_wilcoxon.py` (komentarz w źródle: "for backward
compatibility") — dla testu DWUSTRONNEGO ten znak jest WIĘC ZAWSZE
ujemny, niezależnie od faktycznego kierunku efektu. Zweryfikowane
empirycznie na syntetycznym przykładzie (`event = baseline + 2.0`,
efekt jednoznacznie dodatni): `scipy` zwraca `zstatistic=-7.43`
niezależnie od kolejności argumentów, podczas gdy własna implementacja
(`_wilcoxon_numpy`, suma rang `r_plus` PRZED operacją `abs`) daje
poprawny, kierunkowy `z=+7.43` — identyczny moduł, poprawny znak.
Naprawa: `paired_wilcoxon_test()` liczy `z`/`effect_size_r` WYŁĄCZNIE
własną implementacją; `scipy` (gdy dostępne) używany odtąd TYLKO jako
niezależny cross-check p-wartości (musi się zgadzać co do >=6 miejsc
po przecinku, inaczej `RuntimeError`) — sprawdzone, że cross-check
PRZECHODZI na wszystkich komórkach testu głównego poniżej. Ta poprawka
została wprowadzona PRZED odczytaniem/zinterpretowaniem jakiejkolwiek
liczby z drugiego przebiegu — nie jest to retuning po wyniku, tylko
naprawa zidentyfikowanego błędu mechaniki przed interpretacją.

## 3. Test PRIMARNY: N=3 (cur_L1,L2,L3), baseline vs event, 512 par

### 3.1 `spectral_concentration` (metryka klasyfikująca)

| window | p | r | efekt | mediana baseline | mediana event | status |
|---|---|---|---|---|---|---|
| 128 | 1.39e-85 | +0.866 | duży | 0.5000 | 0.6940 | SUPPORTED |
| 256 | 1.47e-85 | +0.866 | duży | 0.5000 | 0.6918 | SUPPORTED |

**Stabilność znaku**: `r=+0.866(128)→+0.866(256)` — **ZNAK STABILNY**,
niemal identyczna wartość efektu między oknami. **Kierunek ZGODNY z
przewidywaniem a priori** (PREREG §3.4): event > baseline. n=512 par,
0 par odrzuconych (brak NaN, brak remisów zerowych).

**Uwaga o mediannie baseline = dokładnie 0.5000**: to NIE jest
artefakt liczbowy. Dla trzech ZBALANSOWANYCH sinusoid 50 Hz przesuniętych
o 120° (normalny stan przedzwarciowy sieci trójfazowej), macierz
korelacji Pearsona ma teoretycznie dokładnie `corr=cos(120°)=-0.5` poza
przekątną, co daje widmo `{0, 1.5, 1.5}` i `spectral_concentration=
1.5/3=0.5` DOKŁADNIE, niezależnie od amplitudy/szumu. Zaobserwowana
mediana `0.5000000000...` na 512 oknach baseline jest więc SANITY
CHECK, że pipeline poprawnie odczytuje niezaburzony, zbalansowany stan
sieci — nie dowód efektu sam w sobie, ale silny sygnał, że okna
baseline są tym, czym mają być.

### 3.2 `participation_ratio` (diagnostyka, oczekiwane lustro §3.1 dla N=3 tylko częściowo)

| window | p | r | mediana baseline | mediana event |
|---|---|---|---|---|
| 128 | 1.10e-84 | −0.862 | 2.0000 | 1.7569 |
| 256 | 1.39e-77 | −0.824 | 2.0000 | 1.7499 |

Lustrzany znak względem §3.1 (spójne: koncentracja rośnie →
participation ratio maleje), ale — w odróżnieniu od testu N=2 w
domenie łożyskowej — przy N=3 te dwie metryki NIE SĄ matematycznie
zdeterminowane wzajemnie (trzeci stopień swobody `λ_2/λ_3`), więc to
jest częściowo niezależne potwierdzenie tego samego zjawiska, nie
tautologia.

### 3.3 `membrane_spectral_ratio` (sekundarna, brzeg/brzeg, NIE klasyfikująca)

| window | p | r | efekt | status wg progu |
|---|---|---|---|---|
| 128 | 1.28e-04 | +0.169 | mały | NOT_SUPPORTED (\|r\|<0.3) |
| 256 | 8.94e-05 | +0.173 | mały | NOT_SUPPORTED (\|r\|<0.3) |

Istotne statystycznie (n=512 daje moc do wykrycia nawet małych
efektów), znak dodatni i stabilny, ale rozmiar efektu poniżej progu
klasyfikacji (`|r|≥0.3`) — zgodnie z PREREG §5 NIE zmienia
klasyfikacji testu primarnego. Interpretacja: większość sygnału jest w
POZIOMIE koncentracji widma (§3.1), nie w jego zmianie wewnątrz
30-40 ms okna event — podobnie jak w domenie łożyskowej, gdzie
`membrane_spectral_ratio` nie niosło sygnału wcale (tu niesie, ale
słaby).

## 4. Test SEKUNDARNY: N=6 (cur_L1-L3 + vol_L1-L3), baseline vs event, 512 par

### 4.1 `spectral_concentration`

| window | p | r | efekt | mediana baseline | mediana event | status |
|---|---|---|---|---|---|---|
| 128 | 3.65e-85 | +0.864 | duży | 0.5000 | 0.6137 | SUPPORTED |
| 256 | 4.25e-82 | +0.848 | duży | 0.5000 | 0.6139 | SUPPORTED |

**Stabilność znaku**: `r=+0.864(128)→+0.848(256)` — **ZNAK STABILNY**.
Kierunek zgodny z przewidywaniem. Efekt jest nieco SŁABSZY co do
wielkości mediany (event≈0.61 vs 0.69 dla N=3) — spójne z oczekiwaniem:
przy N=6 dodanie trzech kanałów napięcia (które w typowym zwarciu
zmieniają się MNIEJ dramatycznie niż prąd w miejscu zwarcia) rozcieńcza
"czystość" sygnału prądowego, ale NIE niszczy go — koncentracja nadal
rośnie wyraźnie i istotnie ponad baseline `1/6≈0.167` i ponad
zbalansowany stan `0.5`.

### 4.2 `participation_ratio` i `membrane_spectral_ratio` (diagnostyka)

| window | metryka | p | r | efekt |
|---|---|---|---|---|
| 128 | participation_ratio | 2.35e-44 | −0.617 | duży |
| 256 | participation_ratio | 3.17e-34 | −0.539 | duży |
| 128 | membrane_spectral_ratio | 4.53e-26 | +0.467 | średni |
| 256 | membrane_spectral_ratio | 1.05e-09 | +0.270 | mały |

Przy N=6 `membrane_spectral_ratio` niesie WYRAŹNIE silniejszy sygnał
niż przy N=3 (średni efekt na oknie 128, w porównaniu do małego przy
N=3) — ciekawa różnica jakościowa między testem primarnym i
sekundarnym, zgłoszona jako obserwacja eksploracyjna, NIE część
klasyfikacji formalnej (PREREG §5: tylko `spectral_concentration`
klasyfikuje).

## 5. Porównanie N=3 vs N=6

| | N=3 primarny | N=6 sekundarny |
|---|---|---|
| status (oba okna) | SUPPORTED | SUPPORTED |
| r (window=128) | +0.866 | +0.864 |
| r (window=256) | +0.866 | +0.848 |
| znak stabilny | TAK | TAK |
| mediana event (spectral_concentration) | 0.69 | 0.61 |

Oba testy dają JAKOŚCIOWO ten sam wynik: zwarcie zwiększa koncentrację
widmową macierzy korelacji między fazami, efekt ogromny (`|r|>0.8`,
kategoria "duży" na całej siatce), stabilny między oknami i między
wszystkimi 4 typami zwarcia (sprawdzone dodatkowo, §6). Dodanie
napięcia (N=6) osłabia nieco wielkość mediany efektu, ale NIE zmienia
klasyfikacji ani znaku — konstrukcja jest odporna na wybór N w tym
zakresie.

## 6. Kontrola dodatkowa: efekt NIE jest napędzany jednym typem zwarcia

Rozbicie testu głównego (spectral_concentration, oba N, oba okna) po
`sc_type` (4 typy, 128 epizodów każdy, z `labels.csv`):

| sc_type | N=3, window=128, r | N=3, window=256, r | N=6, window=128, r | N=6, window=256, r |
|---|---|---|---|---|
| 0 | +0.868 | +0.868 | +0.837 | +0.702 |
| 1 | +0.868 | +0.868 | +0.868 | +0.868 |
| 2 | +0.868 | +0.868 | +0.868 | +0.868 |
| 3 | +0.868 | +0.867 | +0.867 | +0.865 |

Znak dodatni i wielkość efektu blisko maksymalnej możliwej separacji
rangowej (`r≈0.87` przy n=128 odpowiada niemal doskonałej separacji
baseline/event) na WSZYSTKICH 4 typach zwarcia, na WSZYSTKICH
kombinacjach N/window — efekt nie jest artefaktem jednej klasy
zwarcia dominującej pulę 512 epizodów.

## 7. Wniosek

**Mechanizm poprawnie zachowuje się na kontroli syntetycznej** (§2, 9/9
komórek PASSED, N∈{2,3,6}) — replika walidacji łożyskowej plus nowy
przypadek N=6, bez zmiany kodu.

**Na 512 zamrożonych epizodach PROTECT-90, `spectral_concentration`
odróżnia baseline od event zwarciowego z `p` rzędu `1e-77`–`1e-85`,
efektem "dużym" (`r≈0.85–0.87`) i IDENTYCZNYM znakiem na obu rozmiarach
okna, dla OBU testów (N=3 primarny, N=6 sekundarny) i na WSZYSTKICH 4
typach zwarcia.** Kierunek jest ZGODNY z przewidywaniem a priori
(PREREG §3.4: event > baseline) — w odróżnieniu od domeny łożyskowej,
gdzie identyczny mechanizm dał silny, stabilny efekt w kierunku
PRZECIWNYM do przewidywania. **Formalna klasyfikacja tej sesji:
SUPPORTED dla obu testów (N=3 i N=6)**, zgodnie z kryterium
pre-rejestrowanym w PREREG §5 (`p<0.05`, `|r|≥0.3`, znak stabilny
między oknami, kierunek zgodny z przewidywaniem).

To jest pierwszy wynik w rodzinie `*membrane_bridge*` tego ekosystemu,
gdzie kierunek przewidywania a priori ("silne wspólne zaburzenie →
wyższa koncentracja widma") faktycznie SIĘ POTWIERDZA na realnych
danych, nie tylko daje silny efekt w przeciwną stronę. Jawna
interpretacja fizyczna (post-hoc co do SZCZEGÓŁÓW, ale zgodna z
mechanizmem zapisanym PRZED danymi): zwarcie wprowadza silne,
WSPÓLNE, symultaniczne zaburzenie na wszystkich trzech fazach prądu
danej linii (prąd zwarciowy płynący przez wspólny punkt uszkodzenia),
podczas gdy stan przedzwarciowy jest zbalansowanym, ale NIE identycznym
sygnałem trójfazowym (koncentracja dokładnie `0.5`, nie `1.0`) — różnica
`0.5→0.69` (N=3) / `0.5→0.61` (N=6) odzwierciedla przejście od
"widma zdegenerowanego przez symetrię 120°" do "widma zdominowanego
przez jeden wspólny impuls zwarciowy".

`membrane_spectral_ratio` (brzeg/brzeg) niesie SŁABSZY sygnał niż
poziom koncentracji (§3.3, N=3: efekt "mały", poniżej progu
klasyfikacji) — podobny wzorzec do domeny łożyskowej (informacja jest
głównie w STANIE koncentracji, nie w jej trendzie wewnątrz krótkiego
okna), choć tu przy N=6 efekt tej metryki jest wyraźnie silniejszy
(§4.2, "średni" na oknie 128) — różnica warta osobnej, przyszłej
pre-rejestracji, nie rozstrzygana tutaj.

## 8. Status w ekosystemie

Wynik POZYTYWNY (SUPPORTED) na nowej domenie (sieć elektroenergetyczna,
symulacja EMT PROTECT-90), tym samym, niezmodyfikowanym kodem
matematycznym co domena łożyskowa (`spectral_family.py`) — pierwsza
domena w rodzinie `*membrane_bridge*`, gdzie kierunek przewidywania a
priori się potwierdza. Odróżniony jawnie od `chrono_membrane_bridge`
(domena CWRU, NOT_SUPPORTED co do kierunku mimo silnego efektu) —
inna domena danych, ten sam mechanizm matematyczny, PRZECIWNY wynik co
do zgodności z hipotezą a priori. Błąd znaku `scipy.stats.wilcoxon`
wykryty i naprawiony w tej sesji (§2.1) jest odnotowany jawnie, bo
mógłby (gdyby nie wykryty) dać FAŁSZYWIE negatywny wynik.

## 9. Co zostaje otwarte

- Test na INNYCH lokalizacjach niż `Bus_1_Line_01_02A` (wybór był
  regułą "pierwsza alfabetycznie", nie sprawdzeniem, czy efekt
  utrzymuje się na pozostałych 7) — naturalny następny krok,
  wymagałby osobnej pre-rejestracji (te same dane, więc NIE byłby
  niezależną replikacją).
- Test na `fault_target`/`sc_location` jako zmiennej dodatkowej (czy
  odległość zwarcia od `Bus_1_Line_01_02A` moduluje wielkość efektu) —
  NIE zbadane tutaj.
- Różnica jakościowa `membrane_spectral_ratio` między N=3 (mały efekt)
  i N=6 (średni efekt, §4.2 vs §3.3) — warta osobnej, przyszłej
  pre-rejestracji.
- Błąd znaku scipy (§2.1) sugeruje, że KAŻDE inne miejsce w tym
  ekosystemie używające `scipy.stats.wilcoxon(alternative="two-sided",
  method="approx"/"asymptotic")` z odczytem `.zstatistic` powinno być
  zaudytowane pod tym samym kątem — NIE wykonane w tej sesji (poza
  zakresem), odnotowane jako ryzyko do sprawdzenia.
