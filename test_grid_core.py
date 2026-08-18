"""
test_grid_core.py — testy dla grid_core.py (prymitywy TIMDR).

Kluczowa regresja: `rhythm()` NIE MOŻE zgłaszać fałszywej okresowości na
sygnale bez żadnej realnej struktury cyklicznej (czysty trend + szum) -
to dokładnie ten bug znaleziony empirycznie w pierwszej wersji tej
funkcji napisanej dla tego repo (autokorelacja bez normalizacji per-lag +
dopuszczenie brzegu okna jako "lokalne maksimum" dawały fałszywy,
najkrótszy możliwy okres na każdym gładkim, nieokresowym sygnale).
"""

import numpy as np
import pytest

from grid_core import anomalies, defect, rhythm, resonance, _mad_z


def _rng():
    return np.random.default_rng(42)


# ---------------------------------------------------------------------
# anomalies
# ---------------------------------------------------------------------

def test_anomalies_puste_pasmo_na_czystym_szumie():
    rng = _rng()
    clean = np.full(500, 230.0) + rng.normal(0, 0.5, 500)
    idx = anomalies(clean)
    assert len(idx) < 15  # <3% fałszywych alarmów na czystym gaussowskim szumie


def test_anomalies_lapie_realny_outlier():
    rng = _rng()
    x = np.full(300, 230.0) + rng.normal(0, 0.5, 300)
    x[150] = 300.0  # ewidentny wyrzut
    idx = anomalies(x)
    assert 150 in idx


# ---------------------------------------------------------------------
# defect
# ---------------------------------------------------------------------

def test_defect_brak_falszywych_alarmow_na_plaskim_sygnale():
    flat = np.full(500, 50.0)
    assert len(defect(flat)) == 0


def test_defect_lapie_realny_skok():
    rng = _rng()
    step = np.concatenate([
        np.full(200, 50.0) + rng.normal(0, 0.01, 200),
        np.full(200, 48.0) + rng.normal(0, 0.01, 200),
    ])
    idx = defect(step)
    assert any(195 <= i <= 205 for i in idx)


def test_defect_podloga_na_rzadkich_zmianach():
    """`defect()` jest zaprojektowane pod kanały z realną, nie-zerową
    skalą nominalną (napięcie ~230V, częstotliwość ~50Hz - tak jak w
    timdr_core_finance.py dla cen, nigdy bliskich zeru). Podłoga
    (`min_floor_frac * mediana(|x|)`) skaluje się z tą nominalną
    wartością - na kanale w pobliżu zera (mediana(|x|)≈0) podłoga
    również zapada się do ~0 i nie chroni przed znikomymi ruchami.
    To NIE dotyczy żadnego z czterech kanałów sieci energetycznej w tym
    repo (wszystkie mają realną, nie-zerową skalę), ale jest udokumentowanym
    ograniczeniem tej ogólnej funkcji - test poniżej używa realistycznego
    kanału (częstotliwość w pobliżu 50Hz), nie sztucznego sygnału zerowego."""
    x = np.full(300, 50.0)
    x[150] += 1e-6  # znikomy ruch względem realnej skali ~50Hz
    idx = defect(x, min_floor_frac=1e-2)
    assert len(idx) == 0


def test_defect_podloga_zapada_sie_na_kanale_bliskim_zera_udokumentowane_ograniczenie():
    """Udokumentowane ograniczenie (nie błąd wymagający naprawy w tym
    repo): na kanale, którego typowa wartość jest bliska zeru, podłoga
    progu również zapada się do ~0, więc nawet znikome ruchy mogą zostać
    złapane. Żaden z czterech kanałów TIMDR-Grid-Monitor (napięcie,
    częstotliwość, THD, obciążenie) nie ma takiej charakterystyki."""
    x = np.zeros(300)
    x[150] = 1e-6
    idx = defect(x, min_floor_frac=1e-2)
    assert len(idx) > 0  # potwierdza ograniczenie, nie ukrywa go


# ---------------------------------------------------------------------
# rhythm - regresja fałszywej periodyczności
# ---------------------------------------------------------------------

def test_rhythm_brak_falszywej_periodycznosci_na_czystym_trendzie():
    """Regresja: pierwsza wersja tej funkcji (bez normalizacji per-lag i
    z dopuszczeniem brzegu okna jako lokalnego maksimum) zgłaszała
    fałszywy okres=min_lag na KAŻDYM gładkim, nieokresowym trendzie."""
    rng = _rng()
    n = 2000
    trend_only = np.linspace(1000, 5000, n) + rng.normal(0, 20, n)
    periods, power = rhythm(trend_only)
    assert periods == []
    assert power == 0.0


def test_rhythm_wykrywa_realna_okresowosc_dlugi_okres():
    rng = _rng()
    n = 2000
    t = np.arange(n)
    real_period = 200
    cyclic = 2000 + 800 * np.sin(2 * np.pi * t / real_period) + rng.normal(0, 30, n)
    periods, power = rhythm(cyclic, max_lag=250)
    assert periods, "nie wykryto żadnego okresu"
    closest = min(periods, key=lambda p: abs(p - real_period))
    assert abs(closest - real_period) <= 10
    assert power > 0.9


def test_rhythm_wykrywa_realna_okresowosc_krotki_okres():
    rng = _rng()
    n = 2000
    t = np.arange(n)
    real_period = 50
    cyclic = 2000 + 600 * np.sin(2 * np.pi * t / real_period) + rng.normal(0, 30, n)
    periods, power = rhythm(cyclic)
    closest = min(periods, key=lambda p: abs(p - real_period))
    assert abs(closest - real_period) <= 5


def test_rhythm_pusty_lub_zbyt_krotki_sygnal_nie_wywala_wyjatku():
    assert rhythm(np.array([])) == ([], 0.0)
    assert rhythm(np.array([1.0, 2.0, 3.0])) == ([], 0.0)


def test_rhythm_stala_wartosc_nie_daje_periodycznosci():
    flat = np.full(500, 42.0)
    periods, power = rhythm(flat)
    assert periods == []
    assert power == 0.0


def test_rhythm_gladki_dlugookresowy_sygnal_ze_szumem_nie_daje_falszywych_krotkich_okresow():
    """Regresja: sinus o BARDZO długim okresie (nieusuwany przez zwykły
    detrend liniowy) + drobny szum numeryczny dawał fałszywe 'okresy'
    2-11 próbek z mocą ~0.99 - artefakt gładkości sygnału na krótkich
    opóźnieniach (sąsiednie próbki gładkiej krzywej są z natury podobne),
    NIE prawdziwa cykliczność. Znalezione empirycznie przy testowaniu
    scenariusza demo (dobowy wzorzec obciążenia)."""
    rng = np.random.default_rng(1)
    n = 6000
    t = np.linspace(0, 4 * np.pi, n)  # okres rzeczywisty ~3000 próbek - poza zasięgiem szukania
    smooth_long_period = 0.4 + 0.15 * np.sin(t) + rng.normal(0, 0.01, n)
    periods, power = rhythm(smooth_long_period, max_lag=200)
    assert periods == [], f"fałszywa periodyczność wykryta: {periods}"


# ---------------------------------------------------------------------
# resonance
# ---------------------------------------------------------------------

def test_resonance_wykrywa_jednoczesna_anomalie_wielokanalowa():
    rng = _rng()
    n = 300
    voltage = np.full(n, 230.0) + rng.normal(0, 0.3, n)
    freq = np.full(n, 50.0) + rng.normal(0, 0.02, n)
    voltage[150] = 180.0
    freq[150] = 49.0
    score, strong = resonance({"voltage": voltage, "frequency": freq})
    assert 150 in strong
    assert score[150] == 1.0


def test_resonance_pojedynczy_kanal_nie_osiaga_progu_min_agree():
    rng = _rng()
    n = 300
    voltage = np.full(n, 230.0) + rng.normal(0, 0.3, n)
    freq = np.full(n, 50.0) + rng.normal(0, 0.02, n)
    voltage[150] = 180.0  # tylko JEDEN kanał anomalny
    score, strong = resonance({"voltage": voltage, "frequency": freq}, min_agree=2)
    assert 150 not in strong


def test_resonance_niezgodna_dlugosc_kanalow_rzuca_czytelny_blad():
    with pytest.raises(ValueError, match="długość"):
        resonance({"a": np.zeros(100), "b": np.zeros(50)})
