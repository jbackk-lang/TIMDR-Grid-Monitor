"""
grid_core.py — prymitywy TIMDR dla monitoringu sieci energetycznej
================================================================================
Cztery kanały: napięcie, częstotliwość, THD (zniekształcenia harmoniczne),
obciążenie. Ten moduł dostarcza generyczne prymitywy TIMDR (anomalia/defekt/
rezonans + `rhythm` do wykrywania okresowości) - te same, sprawdzone wzorce
co w `timdr_core_finance.py`/`bio_core.py`/`catalog_core.py` z poprzednich
repozytoriów tego zestawu, zaadaptowane do sygnałów sieci energetycznej.

Różnica względem finansów/EKG: napięcie i częstotliwość mają REALNE,
bezwzględne wartości odniesienia (230V, 50Hz - fizyczne stałe normy
EN 50160), nie tylko lokalną statystykę okna. Dlatego progi tutaj łączą
DWIE rzeczy: (a) bezwzględne limity z normy (nigdy nie przeoczą realnego
przekroczenia, niezależnie od tego, jak "przyzwyczajona" jest sieć do
podwyższonych wartości), (b) adaptacyjną kalibrację z okna (łapie względne
odchylenia od LOKALNEGO zachowania tej konkretnej sieci, nawet jeśli
mieszczą się w normie). Zob. README - to bezpośrednia lekcja z bugu
znalezionego w kodzie referencyjnym: czysto adaptacyjny próg (3× mediana)
na przewlekle podwyższonym THD (mediana=6%) dawał próg 18% - MIJAŁ realne
przekroczenie limitu 8% z normy EN 50160.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# Pomocnicze - te same, zweryfikowane wzorce co w poprzednich repo
# ---------------------------------------------------------------------

def _mad_z(x: np.ndarray) -> np.ndarray:
    """Odporny z-score: (x - mediana) / (1.4826 * MAD), z fallbackiem na
    rozstęp/4 gdy MAD=0 (płaski sygnał) - identyczny wzorzec jak w
    timdr_core_finance.py/bio_core.py."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad > 1e-12:
        scale = 1.4826 * mad
    else:
        spread = np.max(x) - np.min(x)
        scale = spread / 4.0 if spread > 1e-12 else 1.0
    return (x - med) / scale


def _rolling_percentile_spread(x: np.ndarray, window: int, lo: float = 10, hi: float = 90) -> np.ndarray:
    """Rozstęp percentyli w oknie kroczącym, TYLKO wstecz (przyczynowe -
    bez zaglądania w przyszłość) - ten sam wzorzec co `defect()` w
    timdr_core_finance.py."""
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        seg = x[start:i + 1]
        if len(seg) < 3:
            out[i] = 0.0
        else:
            out[i] = np.percentile(seg, hi) - np.percentile(seg, lo)
    return out


def anomalies(x: np.ndarray, factor: float = 3.0) -> np.ndarray:
    """Indeksy próbek odstających wg odpornego z-score (MAD-z)."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.array([], dtype=int)
    z = _mad_z(x)
    return np.where(np.abs(z) > factor)[0]


def defect(x: np.ndarray, window: int = 20, jump_factor: float = 3.0,
           min_floor_frac: float = 1e-4) -> np.ndarray:
    """Nagłe skoki między kolejnymi próbkami, względem rozstępu RÓŻNIC w
    oknie wstecznym (nie rozstępu poziomów - to była udokumentowana
    pułapka w timdr_core_finance.py). Podłoga zapobiega zapadaniu się
    progu do zera na płaskich odcinkach."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return np.array([], dtype=int)
    diffs = np.abs(np.diff(x))
    diffs_padded = np.concatenate([[0.0], diffs])
    spread = _rolling_percentile_spread(diffs_padded, window)
    floor = min_floor_frac * (np.median(np.abs(x)) + 1e-9)
    threshold = np.maximum(jump_factor * spread, floor)
    idx = np.where(diffs_padded > threshold)[0]
    return idx[idx > 0]


def rhythm(x: np.ndarray, max_lag: int = 60, power_thresh: float = 0.3, dip_frac: float = 0.5) -> tuple:
    """
    Wykrywanie okresowości - punkt wyjścia to `rhythm()` z
    `timdr_core_finance.py` (zweryfikowany, sprawdzony w kilku innych
    repo tego zestawu wzorzec: pełny detrend + autokorelacja
    NORMALIZOWANA per-lag + filtr tylko ŚCIŚLE WEWNĘTRZNYCH lokalnych
    maksimów, lag=2..max_lag-1, nigdy brzeg okna).

    DODATKOWA NAPRAWA znaleziona przy testowaniu TEGO repo (na danych
    energetycznych, nie finansowych): sygnał gładki i wolnozmienny (np.
    dobowy wzorzec obciążenia - sinus o BARDZO długim okresie, nie
    usuwany przez zwykły detrend liniowy) generuje autokorelację, która
    jest wysoka i PRAWIE PŁASKA na krótkich opóźnieniach (bo sąsiednie
    próbki gładkiej krzywej są z natury do siebie podobne) - drobny szum
    numeryczny na tym wysokim "plateau" tworzy przypadkowe, malutkie
    zafalowania, które formalnie spełniają test "lokalne maksimum", mimo
    że nie reprezentują żadnej prawdziwej struktury okresowej (empirycznie
    potwierdzone: gładki sinus o okresie ~3000 próbek + szum dawał
    fałszywe "okresy" 2-11 próbek z mocą ~0.99). Naprawiono: kandydat na
    okres jest przyjmowany TYLKO jeśli autokorelacja zdążyła realnie
    OPAŚĆ (do mniej niż `dip_frac` jego własnej wartości) w którymś
    wcześniejszym opóźnieniu, zanim ponownie wzrosła do tego szczytu -
    odróżnia to "prawdziwy cykl" (spadek, potem wzrost) od "wciąż jesteśmy
    na początkowym, gładkim zboczu opadania korelacji blisko lag=0".

    Zwraca (periods, power) - periods: lista okresów (w próbkach) z
    lokalnym maksimum autokorelacji powyżej `power_thresh`, posortowana
    malejąco po sile; power: siła najsilniejszego z nich (0..1).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 8:
        return [], 0.0

    idx = np.arange(n, dtype=float)
    coeffs = np.polyfit(idx, x, 1)
    detrended = x - np.polyval(coeffs, idx)

    std = np.std(detrended)
    if std == 0:
        return [], 0.0

    max_lag = min(max_lag, n - 2)
    if max_lag < 2:
        return [], 0.0

    acf = np.zeros(max_lag + 1)
    for lag in range(1, max_lag + 1):
        a, b = detrended[:-lag], detrended[lag:]
        denom = np.std(a) * np.std(b) * len(a)
        acf[lag] = 0.0 if denom == 0 else np.sum(a * b) / denom

    peaks = []
    running_min = acf[1]
    for lag in range(2, max_lag):
        is_local_max = acf[lag] > acf[lag - 1] and acf[lag] > acf[lag + 1] and acf[lag] > power_thresh
        has_real_dip = running_min < dip_frac * acf[lag]
        if is_local_max and has_real_dip:
            peaks.append((lag, float(acf[lag])))
        running_min = min(running_min, acf[lag])

    if not peaks:
        return [], 0.0
    peaks.sort(key=lambda p: -p[1])
    return [p[0] for p in peaks], peaks[0][1]


def resonance(signals: dict, factor: float = 3.0, min_agree: int = 2) -> tuple:
    """
    Rezonans międzykanałowy: dla każdej próbki liczy, ile z podanych
    kanałów jednocześnie zgłasza anomalię (MAD-z > factor). Zwraca
    (score_0_do_1, strong_idx) - score = ułamek kanałów zgadzających się
    per próbka, strong_idx = indeksy, gdzie zgodziło się >= min_agree
    kanałów jednocześnie (silniejszy, bardziej wiarygodny sygnał niż
    pojedyncza anomalia w jednym kanale - ten sam wzorzec co `resonance()`
    w timdr_core_finance.py, tu uogólniony na DOWOLNĄ liczbę kanałów
    zamiast zawsze-trzech wewnętrznych sprawdzeń).
    """
    names = list(signals.keys())
    if not names:
        return np.array([]), np.array([], dtype=int)
    n = len(next(iter(signals.values())))
    flags = np.zeros((len(names), n), dtype=bool)
    for i, name in enumerate(names):
        arr = np.asarray(signals[name], dtype=float)
        if len(arr) != n:
            raise ValueError(f"resonance(): kanał '{name}' ma inną długość ({len(arr)}) niż pozostałe ({n})")
        z = _mad_z(arr)
        flags[i] = np.abs(z) > factor
    agree_count = flags.sum(axis=0)
    score = agree_count / len(names)
    strong_idx = np.where(agree_count >= min_agree)[0]
    return score, strong_idx
