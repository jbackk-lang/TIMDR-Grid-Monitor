"""meta_adapter.py -- adapter TimdrEnergySignals (4 kanaly: napiecie/
czestotliwosc/THD/obciazenie) -> MetaState (integracja z
TIMDR-META-DYNAMICS), piata realna integracja formalizmu
Lambda-tau-rho-J w tym ekosystemie (pierwsza: finansowa w
analizator-gieldowy-v3, druga: pogodowa w Synoptyk-v3, trzecia: sejsmiczna
w TIMDR-Earthquake-Core, czwarta: wibracja lozysk w
TIMDR-Industrial-Predict).

===========================================================================
DLACZEGO ARCHITEKTURA JEDNOSLADOWA (jak sejsmika), NIE DWUSCIEZKOWA
(jak lozyska):

Uszkodzenie lozyska (TIMDR-Industrial-Predict/bearing_meta_adapter.py) to
STAN STALY przez caly zapis testowy - nie ma tam "spokojnego okresu przed
zdarzeniem" wewnatrz jednego sladu, stad architektura dwoch OSOBNYCH
nagran (referencja+test). Zdarzenia sieciowe sa inne: mikro-zanik
napiecia, przeciazenie, anomalia harmoniczna - to wszystko PRZEJSCIA W
SRODKU jednego ciaglego zapisu (spokoj -> zdarzenie -> powrot), dokladnie
jak mainshock trzesienia ziemi w TIMDR-Earthquake-Core. Dlatego ten
adapter uzywa TEJ SAMEJ, jednosladowej architektury co adapter sejsmiczny
(kalibracja `rolling_history_seconds`), NIE architektury lozyskowej.

REUZYCIE (nie duplikacja czwarty raz), TERAZ PRZEZ WENDOROWANIE, NIE
SIBLING-IMPORT (zmiana 2026-09-10): `compute_global_thresholds()`/
`window_to_meta_state()`/`build_meta_series_from_waveform()` w
TIMDR-Earthquake-Core/meta_adapter.py NIE sa w swojej logice specyficzne
dla sejsmiki - dzialaja na dowolnej parze (t,s) i przyjmuja `core` jako
parametr. Ten plik pierwotnie sibling-importowal je WPROST z
TIMDR-Earthquake-Core; teraz uzywa lokalnej, zwendorowanej kopii
(`_vendor_seismic_meta_windowing.py`, patrz jej naglowek) zamiast
wymagac obecnosci innego repo jako folderu-siostry na dysku - powod:
wyrazna prosba, zeby repozytoria kodu byly niezalezne od siebie. Kod
pozostaje reuzyty (nie przepisany od zera piaty raz w tym ekosystemie:
pierwszy raz analizator-gieldowy-v3, potem Synoptyk-v3, TIMDR-Earthquake-Core
samo, TIMDR-Industrial-Predict przeniosl tylko flow()/trm() - TEN plik
idzie o krok dalej i reuzywa CALA warstwe okienkowania, nie tylko flow/
trm) - zmienil sie tylko MECHANIZM reuzycia (kopia zamiast sibling-importu
w czasie wykonania), nie sama matematyka. Wynikowa klasa
`SeismicMetaResult` jest uzywana wprost, bez zmiany nazwy - ksztalt
(window_starts/states/M_series/phases/trigger) jest identyczny
niezaleznie od domeny, przemianowanie byloby czysto kosmetyczne.

===========================================================================
GENEZA (2026-09-08) - jedna nieudana proba PRZED tym, co ponizej,
udokumentowana zamiast ukryta:

PROBA 1 (WINDOW_SECONDS=0.5s/50 probek, ROLLING_HISTORY_SECONDS=5.0s,
K_NEIGHBORS=3 - dobrane PRZED dotknieciem danych, na podstawie analizy
czasu trwania najkrotszego zdarzenia w demo_generator.py, patrz zastrzezenie
#1 nizej): KONTROLA NEGATYWNA NIE PRZESZLA. Na scenariuszu `normalny`
(zdrowa siec, ZERO wstrzknietych zdarzen) wszystkie cztery kanaly wyszly
w >=95% okien "przejsciowa" albo "krytyczna", zamiast oczekiwanego
"stabilna". Przyczyna (znaleziona empirycznie, nie zgadywaniem):
Lambda (FFT high-freq fraction) na oknie 50 probek czystego szumu
gaussowskiego ma OGROMNA wariancje probkowania miedzy sasiednimi oknami
(np. 0.63 -> 0.36 -> 0.45 -> 0.54 w kolejnych oknach `normalny`/voltage) -
50 probek to za malo binow FFT, zeby usrednic szum estymacji. Ten sam
mechanizm dotyczy rho (fraction anomalnych probek w oknie): przy tak
krotkim oknie pojedyncze przypadkowe probki bliskie progu 3.5*MAD
przeskakuja go i znikaja miedzy sasiednimi oknami, dajac delta_rho rzedu
0.1-0.24 CZYSTO Z SZUMU. Podzielone przez dt=WINDOW_SECONDS=0.5 (male dt
WZMACNIA kazda zmiane stanu w M=delta_S/dt), to wystarczylo, zeby
magnitude(M) regularnie przekraczalo prog 0.1 na czystym szumie -
analogiczny problem co Synoptyk-v3 meta_adapter V1 (tam odwrotny
kierunek: surowe liczby zdominowaly sume i dalo "krytyczna" wszedzie;
tu zbyt krotkie okno statystyczne dalo szum wiekszy niz sygnal).

PROBA 2 (AKTUALNA) - WIEKSZE OKNO STATYSTYCZNE: WINDOW_SECONDS=2.0s (200
probek), ROLLING_HISTORY_SECONDS=10.0s (100 probek historii - 5 okien).
200 probek na FFT/rho daje wystarczajaco stabilna statystyke: kontrola
negatywna (`normalny`) wychodzi 100% "stabilna" na voltage, >=95% na
frequency/harmonics; `load` pozostaje czesciowo niestabilny nawet w
kontroli negatywnej (patrz zastrzezenie #5 - to REALNA wlasciwosc kanalu
load, nie blad kalibracji). Zapłacona cena: rolling_history_seconds=10s
"zjada" pierwsze 10s kazdego 60-80s sladu (okna nie moga byc policzone
bez pelnej historii sprzed nich) - por. zastrzezenie #6.

===========================================================================
PRE-REJESTRACJA MAPOWANIA V2 (ustalone PRZED ponownym uruchomieniem po
PROBIE 1 - protokol numerologii/formalizmu, skill timdr-signal-framework):

Cztery kanaly (`voltage`, `frequency`, `harmonics`, `load`) sa
analizowane NIEZALEZNIE, kazdy jako wlasny slad (t,s) - TA SAMA
matematyka co adapter sejsmiczny (Lambda=high-freq fraction FFT okna,
tau=flow vs prog globalny, rho=anomalie vs prog globalny, J=twist vs
prog globalny), z `rolling_history_seconds` jako DOMYSLNYM trybem
kalibracji (nie `calibration_end`/pelny-slad) - Earthquake-Core samo
ustalilo (patrz jego wlasny plik, zastrzezenie #7), ze to najlepszy z
trzech wariantow do ciaglej, przyczynowej detekcji, wiec ten plik
dziedziczy ta rekomendacje zamiast ponownie testowac wszystkie trzy.

  SAMPLE_RATE_HZ = 100.0 Hz (dokladnie jak demo_generator.py - polowkowe
                    odswiezanie RMS wg IEC 61000-4-30 przy 50Hz).
  WINDOW_SECONDS = 2.0s (200 probek/okno) - patrz PROBA 1/2 wyzej:
                    najmniejsze okno, przy ktorym kontrola negatywna
                    faktycznie przechodzi na wiekszosci kanalow.
  ROLLING_HISTORY_SECONDS = 10.0s (5 okien trailing historii).
  K_NEIGHBORS = 3 (NIE domyslne 8 z TIMDR_EarthquakeCore) - najkrotsze
                    zdarzenie w demo_generator.py (mikrozanik, 2 probki =
                    20ms) jest krotsze niz domyslne okno k=8 (80ms przy
                    100Hz) - ten sam blad lokalnosci, ktory
                    bearing_meta_adapter.py juz raz udokumentowal
                    (zastrzezenie #6 tamtego pliku). k=3 (30ms) jest
                    krotsze niz najkrotszy mikrozanik (20ms).

UCZCIWE ZASTRZEZENIA:
  1. K_NEIGHBORS=3 i wstepny wybor WINDOW_SECONDS=0.5s (PROBA 1) zostaly
     dobrane PRZED uruchomieniem, na podstawie analizy czasu trwania
     najkrotszego zdarzenia w demo_generator.py - nie po zobaczeniu
     zlego wyniku. WINDOW_SECONDS=2.0s (PROBA 2) BYLO wyborem PO
     zobaczeniu, ze PROBA 1 nie przechodzi kontroli negatywnej - jawnie
     oznaczone jako takie (nie ukrywane jako "zawsze taki byl plan").
     Zaden z tych parametrow nie zostal zweryfikowany na realnych
     danych z licznika/PLC (device_client.py jest szkieletem, swiadomie
     niezaimplementowanym).
  2. WYNIK NA SCENARIUSZACH demo_generator.py (WINDOW_SECONDS=2.0,
     ROLLING_HISTORY_SECONDS=10.0, ~23 okna/kanal na 60s sladzie):
       - `mikrozanik` (voltage): WYRAZNIE WYKRYTE - 7/23 okien
         "przejsciowa" (0/23 w kontroli negatywnej), dokladnie w oknach
         otaczajacych trzy wstrzykniete zdarzenia (t=12s/30s/48s) -
         zweryfikowane window-po-window, nie tylko zliczeniem.
       - `cykliczne_zaklocenia` (load): NIE WYKRYTE - 23/23 "stabilna",
         DOKLADNIE JAK PRZEWIDZIANO PRZED URUCHOMIENIEM (patrz #3):
         M-operator wykrywa ZMIANE, nie STAN OKRESOWY.
       - `przeciazenie` (load, voltage): SLABO/NIEJEDNOZNACZNIE
         WYKRYTE - load pokazuje 6/23 "przejsciowa", ale kontrola
         negatywna (load w `normalny`) juz sama z siebie ma 8/23
         "przejsciowa" (patrz zastrzezenie #5) - przeciazenie NIE
         wystaje ponad wlasny szum tla tego kanalu w tym pomiarze.
       - `anomalia_harmoniczna` (harmonics): UMIARKOWANIE WYKRYTE -
         5/23 "przejsciowa" vs ~1/23 w kontroli negatywnej - realna,
         ale nie ostra roznica.
     Uczciwy wniosek: adapter dziala DOBRZE dla zdarzen typu
     "krotki, ostry impuls na tle stabilnego kanalu" (mikrozanik), SLABO
     dla zdarzen typu "dlugotrwaly przesuniety poziom na kanale, ktory
     juz ma wlasna, nietrywialna dynamike" (przeciazenie na load) - to
     nie jest uniwersalne rozwiazanie dla wszystkich czterech typow
     zdarzen z README tego repo.
  3. `cykliczne_zaklocenia` (okresowe wahania obciazenia, NIE
     jednorazowe przejscie) jest architektonicznie NIEDOPASOWANE do
     M-operatora (wykrywa ZMIANE miedzy oknami, nie STAN OKRESOWY) -
     analogiczny problem do STANU STALEGO w lozyskach (patrz
     bearing_meta_adapter.py PROBA 2). To zostalo PRZEWIDZIANE PRZED
     uruchomieniem (nie odkryte post-hoc) i POTWIERDZONE empirycznie
     (#2 wyzej) - `grid_core.rhythm()` (juz istniejace w tym repo) jest
     wlasciwym narzedziem do tego konkretnego zjawiska, nie ten adapter.
  4. Cztery kanaly sa analizowane NIEZALEZNIE (cztery oddzielne
     SeismicMetaResult) - ten plik NIE buduje jednego, zagregowanego
     "energii stanu E(t)", ani nie uzywa `grid_core.resonance()`
     (koincydencja miedzy kanalami) do laczenia ich w jeden wynik. To
     swiadomy wybor zakresu na start - agregacja miedzykanalowa
     pozostaje otwartym watkiem.
  5. Kanal `load` ma WLASNA, nietrywialna dynamike nawet w kontroli
     negatywnej (dobowy wzorzec `0.4 + 0.15*sin(...)` w
     demo_generator.py, patrz tez ostrzezenie w grid_core.py::rhythm()
     o gladkich, wolnozmiennych sygnalach) - to podnosi bazowy poziom
     tau/rho na tym kanale nawet bez zadnej anomalii, co CZESCIOWO
     tlumaczy zastrzezenie #2 (przeciazenie na load slabo odrozniane od
     wlasnego szumu tla kanalu). Kanaly voltage/frequency/harmonics sa w
     demo_generator.py plaskie (szum bialy bez trendu) poza wstrzykniete
     zdarzenia - stad ich kontrola negatywna jest czystsza.
  6. ROLLING_HISTORY_SECONDS=10.0s na sladzie 60-80s "zjada" pierwsze
     10s (okna bez pelnej trailing historii sa pomijane, patrz docstring
     `build_meta_series_from_waveform` tryb 3) - to az ~15-17% dlugosci
     sladu bez analizy, wiecej niz analogiczny stosunek w
     TIMDR-Earthquake-Core (30s/360s = 8%). Przy realnym, ciaglym
     monitoringu (nie jednorazowej probce demo) to nie jest problem
     (system po prostu "rozgrzewa sie" raz na starcie), ale przy
     krotkich, jednorazowych analizach (jak testy tutaj) to realny koszt.
===========================================================================
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid_monitor import TimdrEnergySignals

# ZWENDOROWANE 2026-09-10 (patrz nagłówki plików `_vendor_*.py` w tym
# repo dla pełnego uzasadnienia): wcześniej ten moduł ładował
# TIMDR-META-DYNAMICS i TIMDR-Earthquake-Core przez sys.path
# sibling-import z folderów-sióstr na dysku. Zamienione na lokalne,
# zwendorowane kopie, żeby to repo działało samodzielnie po sklonowaniu
# WYŁĄCZNIE siebie (decyzja na wyraźną prośbę: "repozytoria kodu mają
# być niezależne od siebie"). Zachowanie/matematyka bez zmian.
from _vendor_timdr_meta_dynamics_core import MetaState, MetaOperatorM
from _vendor_timdr_core_earthquake import TIMDR_EarthquakeCore
from _vendor_seismic_meta_windowing import SeismicMetaResult, build_meta_series_from_waveform

SAMPLE_RATE_HZ = 100.0
WINDOW_SECONDS = 2.0
ROLLING_HISTORY_SECONDS = 10.0
K_NEIGHBORS = 3

CHANNEL_NAMES = ("voltage", "frequency", "harmonics", "load")


@dataclass
class GridMetaResult:
    """Cztery niezalezne wyniki, jeden per kanal - patrz zastrzezenie #3
    w naglowku modulu (brak agregacji miedzykanalowej tutaj)."""
    voltage: SeismicMetaResult
    frequency: SeismicMetaResult
    harmonics: SeismicMetaResult
    load: SeismicMetaResult

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in CHANNEL_NAMES}


def build_grid_meta_series(
    signals: TimdrEnergySignals,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    window_seconds: float = WINDOW_SECONDS,
    rolling_history_seconds: float | None = ROLLING_HISTORY_SECONDS,
    k_neighbors: int = K_NEIGHBORS,
) -> GridMetaResult:
    """Buduje MetaState-serie NIEZALEZNIE dla kazdego z czterech kanalow
    `signals`. `t` jest wyprowadzone z `sample_rate_hz` (TimdrEnergySignals
    nie niesie wlasnej osi czasu - wszystkie cztery kanaly maja WSPOLNA,
    rownomierna os czasu z zalozenia, patrz demo_generator.py)."""
    n = len(signals.voltage)
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    core = TIMDR_EarthquakeCore(k_neighbors=k_neighbors)

    results = {}
    for name in CHANNEL_NAMES:
        s = getattr(signals, name)
        results[name] = build_meta_series_from_waveform(
            t, s,
            window_seconds=window_seconds,
            core=core,
            rolling_history_seconds=rolling_history_seconds,
        )

    return GridMetaResult(**results)
