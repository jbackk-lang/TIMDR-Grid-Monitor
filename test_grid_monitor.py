"""
test_grid_monitor.py — testy dla TimdrEnergyMonitor, w tym regresje
CZTERECH błędów znalezionych w szkicu dostarczonym przez użytkownika
(patrz grid_monitor.py, docstring modułu, i README.md).
"""

import numpy as np
import pytest

from grid_monitor import TimdrEnergySignals, TimdrEnergyEvents, TimdrEnergyMonitor


def _rng():
    return np.random.default_rng(42)


def _base_signals(n=2000, load=None, voltage=None, frequency=None, harmonics=None):
    rng = _rng()
    return TimdrEnergySignals(
        voltage=voltage if voltage is not None else np.full(n, 230.0) + rng.normal(0, 0.3, n),
        frequency=frequency if frequency is not None else np.full(n, 50.0) + rng.normal(0, 0.02, n),
        harmonics=harmonics if harmonics is not None else np.abs(rng.normal(2.0, 0.3, n)),
        load=load if load is not None else 2000 + rng.normal(0, 50, n),
    )


# ---------------------------------------------------------------------
# TimdrEnergySignals - walidacja schematu
# ---------------------------------------------------------------------

def test_signals_niezgodna_dlugosc_kanalow_rzuca_blad():
    with pytest.raises(ValueError, match="różne długości"):
        TimdrEnergySignals(voltage=np.zeros(100), frequency=np.zeros(100),
                            harmonics=np.zeros(100), load=np.zeros(50))


# ---------------------------------------------------------------------
# Regresja bug 1: przeciążenia względem mocy znamionowej, nie max(okna)
# ---------------------------------------------------------------------

def test_regresja_overload_nie_flaguje_zdrowego_profilu():
    """W szkicu użytkownika: próg = 90% z max(load) W TYM OKNIE - na
    zdrowym profilu (26% mocy znamionowej) dawało ~21% fałszywych
    'przeciążeń'. Tu: 0."""
    rng = _rng()
    n = 2000
    rated = 10000.0
    load = 2000 + 500 * np.sin(np.linspace(0, 20 * np.pi, n)) + rng.normal(0, 50, n)
    sig = _base_signals(n, load=load)
    mon = TimdrEnergyMonitor(rated_load=rated)
    ev = mon.analyze(sig, sample_rate_hz=1000.0)
    assert len(ev.overloads) == 0


def test_regresja_overload_wykrywa_realne_przeciazenie():
    rng = _rng()
    n = 2000
    rated = 10000.0
    load = 2000 + rng.normal(0, 50, n)
    load[500:520] = 9500.0
    sig = _base_signals(n, load=load)
    mon = TimdrEnergyMonitor(rated_load=rated)
    ev = mon.analyze(sig, sample_rate_hz=1000.0)
    flagged_idx = {i for i, _, _ in ev.overloads}
    assert len(flagged_idx & set(range(500, 520))) >= 15


def test_overload_bez_rated_load_rzuca_jawny_blad():
    """NIGDY cichy fallback na złamaną heurystykę - jawny błąd zamiast
    tego, jeśli 'rated_load' nie zostało podane."""
    sig = _base_signals(200)
    mon = TimdrEnergyMonitor()  # brak rated_load
    with pytest.raises(ValueError, match="rated_load"):
        mon.analyze(sig, sample_rate_hz=1000.0)


# ---------------------------------------------------------------------
# Mikro-zaniki napięcia
# ---------------------------------------------------------------------

def test_micro_outage_wykrywa_spadek_napiecia():
    rng = _rng()
    n = 1000
    voltage = np.full(n, 230.0) + rng.normal(0, 0.3, n)
    voltage[300:320] = 80.0  # spadek < 50% nominalnej (115V)
    sig = _base_signals(n, voltage=voltage)
    mon = TimdrEnergyMonitor(rated_load=10000.0)
    ev = mon.analyze(sig, sample_rate_hz=1000.0)
    assert len(ev.micro_outages) >= 1
    start, duration_ms = ev.micro_outages[0]
    assert 295 <= start <= 305
    assert duration_ms >= 15  # ~20ms przy 1kHz


def test_micro_outage_ignoruje_zbyt_krotkie_zaniki():
    rng = _rng()
    n = 1000
    voltage = np.full(n, 230.0) + rng.normal(0, 0.3, n)
    voltage[300] = 80.0  # tylko 1 próbka = 1ms przy 1kHz
    sig = _base_signals(n, voltage=voltage)
    mon = TimdrEnergyMonitor(rated_load=10000.0, micro_outage_min_ms=10)
    ev = mon.analyze(sig, sample_rate_hz=1000.0)
    assert len(ev.micro_outages) == 0


# ---------------------------------------------------------------------
# Regresja bug 3: harmoniczne - dualny próg (norma + adaptacyjny)
# ---------------------------------------------------------------------

def test_regresja_harmonic_wykrywa_przekroczenie_normy_mimo_podwyzszonego_tla():
    """W szkicu: próg WYŁĄCZNIE adaptacyjny (3x mediana) - na sieci z
    przewlekle podwyższonym tłem (median=6%) próg (18%) PRZEOCZAŁ
    wstrzyknięte przekroczenie normy EN50160 (9.5% > limit 8%)."""
    rng = _rng()
    n = 2000
    harmonics = np.abs(rng.normal(6.0, 0.5, n))
    harmonics[250] = 9.5
    sig = _base_signals(n, harmonics=harmonics)
    mon = TimdrEnergyMonitor(rated_load=10000.0)
    ev = mon.analyze(sig, sample_rate_hz=1000.0)
    found = [e for e in ev.harmonic_anomalies if e[0] == 250]
    assert len(found) == 1
    assert found[0][2] in ("limit_normy", "oba")


def test_harmonic_zdrowa_siec_bez_falszywych_alarmow():
    rng = _rng()
    n = 1000
    harmonics = np.abs(rng.normal(2.0, 0.3, n))  # zdrowo, daleko od limitu 8%
    sig = _base_signals(n, harmonics=harmonics)
    mon = TimdrEnergyMonitor(rated_load=10000.0)
    ev = mon.analyze(sig, sample_rate_hz=1000.0)
    assert len(ev.harmonic_anomalies) < 20  # nieliczne statystyczne odchylenia OK, powódź - nie


# ---------------------------------------------------------------------
# Regresja bug 2: częstotliwość faktycznie analizowana
# ---------------------------------------------------------------------

def test_regresja_frequency_wczesniej_zupelnie_nieuzywane_teraz_wykrywa():
    rng = _rng()
    n = 2000
    freq = np.full(n, 50.0) + rng.normal(0, 0.02, n)
    freq[1000:1010] = 52.5  # krytyczne, >4% od 50Hz
    sig = _base_signals(n, frequency=freq)
    mon = TimdrEnergyMonitor(rated_load=10000.0)
    ev = mon.analyze(sig, sample_rate_hz=1000.0)
    critical = [e for e in ev.frequency_anomalies if e[2] == "krytyczne"]
    assert len(critical) >= 5


def test_frequency_zdrowa_siec_bez_alarmow():
    rng = _rng()
    n = 1000
    freq = np.full(n, 50.0) + rng.normal(0, 0.02, n)
    sig = _base_signals(n, frequency=freq)
    mon = TimdrEnergyMonitor(rated_load=10000.0)
    ev = mon.analyze(sig, sample_rate_hz=1000.0)
    assert len(ev.frequency_anomalies) == 0


# ---------------------------------------------------------------------
# Regresja bug 4: cykliczne zakłócenia - detrend + lokalne maksima
# ---------------------------------------------------------------------

def test_regresja_cyclic_brak_falszywej_periodycznosci_na_trendzie():
    rng = _rng()
    n = 2000
    load = np.linspace(1000, 5000, n) + rng.normal(0, 20, n)
    sig = _base_signals(n, load=load)
    mon = TimdrEnergyMonitor(rated_load=10000.0)
    ev = mon.analyze(sig, sample_rate_hz=1000.0)
    assert ev.cyclic_disturbances == []


def test_regresja_cyclic_wykrywa_realna_okresowosc_obciazenia():
    rng = _rng()
    n = 2000
    t = np.arange(n)
    load = 2000 + 800 * np.sin(2 * np.pi * t / 150) + rng.normal(0, 30, n)
    sig = _base_signals(n, load=load)
    mon = TimdrEnergyMonitor(rated_load=10000.0)
    ev = mon.analyze(sig, sample_rate_hz=1000.0)
    load_periods = [p for ch, p, pw in ev.cyclic_disturbances if ch == "load"]
    assert any(abs(p - 150) <= 10 for p in load_periods)


def test_cyclic_wykrywa_nawracajace_zdarzenia_niezaleznie_od_load():
    """Cykliczność samych ZDARZEŃ (np. THD skacze co ~40 próbek), nawet
    jeśli load samo w sobie jest płaskie/nieokresowe."""
    rng = _rng()
    n = 2000
    harmonics = np.abs(rng.normal(2.0, 0.3, n))
    for i in range(0, n, 40):
        if i < n:
            harmonics[i] = 9.0  # regularne przekroczenie normy co 40 próbek
    load = np.full(n, 2000.0) + rng.normal(0, 10, n)  # load płaskie, nieokresowe
    sig = _base_signals(n, load=load, harmonics=harmonics)
    mon = TimdrEnergyMonitor(rated_load=10000.0, cyclic_max_lag=100)
    ev = mon.analyze(sig, sample_rate_hz=1000.0)
    event_periods = [p for ch, p, pw in ev.cyclic_disturbances if ch == "zdarzenia"]
    assert any(abs(p - 40) <= 5 for p in event_periods)


# ---------------------------------------------------------------------
# analyze() - integracja pełnego przebiegu
# ---------------------------------------------------------------------

def test_analyze_zdrowa_siec_bez_zadnych_zdarzen():
    sig = _base_signals(1000)
    mon = TimdrEnergyMonitor(rated_load=10000.0)
    ev = mon.analyze(sig, sample_rate_hz=1000.0)
    assert isinstance(ev, TimdrEnergyEvents)
    assert ev.overloads == []
    assert ev.micro_outages == []
    assert ev.cyclic_disturbances == []
