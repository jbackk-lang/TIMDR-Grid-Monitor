"""Testy meta_adapter.py -- adapter TimdrEnergySignals -> MetaState.

Dwie grupy testow:
1. Syntetyczne, szybkie (male okna/slady) - sprawdzaja ksztalt wynikow,
   walidacje wejscia i podstawowe kontrole pozytywna/negatywna na
   krotkich, celowo skonstruowanych sladach.
2. Na scenariuszach demo_generator.py (WOLNIEJSZE - realne parametry
   WINDOW_SECONDS=2.0/ROLLING_HISTORY_SECONDS=10.0 na 60-80s sladach,
   patrz PROBA 1/2 w naglowku meta_adapter.py) - potwierdzaja REALNE,
   udokumentowane w naglowku zachowanie: mikrozanik wykrywany wyraznie,
   cykliczne_zaklocenia NIE wykrywane (architektoniczne, przewidziane
   z gory), przeciazenie slabo/niejednoznacznie.
"""
import numpy as np
import pytest

from meta_adapter import (
    build_grid_meta_series,
    GridMetaResult,
    CHANNEL_NAMES,
    MetaOperatorM,
)


# ---------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------

def _flat_signals(n=2000, seed=0):
    """Cztery plaskie, niezalezne kanaly czystego szumu gaussowskiego -
    kontrola negatywna: zero zdarzen, zadnej struktury. Szumy per-kanal
    dobrane tak, jak w demo_generator.py::_base_channels (te same
    realistyczne skale EN 50160, NIE dowolny jeden wspolny 'sigma' -
    empirycznie sprawdzone, ze jeden generyczny sigma dawal niestabilna,
    zalezna-od-ziarna kontrole negatywna na kanale frequency, patrz
    historia commitow tego pliku)."""
    from grid_monitor import TimdrEnergySignals
    rng = np.random.default_rng(seed)
    return TimdrEnergySignals(
        voltage=230.0 + rng.normal(0, 0.8, n),
        frequency=50.0 + rng.normal(0, 0.02, n),
        harmonics=np.abs(rng.normal(2.0, 0.4, n)),
        load=5000.0 + rng.normal(0, 50.0, n),
    )


def _voltage_dip_signals(n=2000, seed=1, dip_start=1000, dip_len=5):
    """Jak _flat_signals, ale z jednym, krotkim, glebokim zapadem
    napiecia (jak mikrozanik) - kontrola pozytywna dla kanalu voltage."""
    sig = _flat_signals(n=n, seed=seed)
    sig.voltage[dip_start:dip_start + dip_len] = 230.0 * 0.2
    return sig


# ---------------------------------------------------------------------
# Walidacja ksztaltu/wejscia (szybkie, male okna)
# ---------------------------------------------------------------------

def test_build_grid_meta_series_returns_all_four_channels():
    sig = _flat_signals(n=2000)
    res = build_grid_meta_series(sig, window_seconds=1.0, rolling_history_seconds=5.0)
    assert isinstance(res, GridMetaResult)
    for name in CHANNEL_NAMES:
        assert hasattr(res, name)


def test_as_dict_matches_channel_names():
    sig = _flat_signals(n=2000)
    res = build_grid_meta_series(sig, window_seconds=1.0, rolling_history_seconds=5.0)
    d = res.as_dict()
    assert set(d.keys()) == set(CHANNEL_NAMES)


def test_each_channel_result_has_states_and_phases():
    sig = _flat_signals(n=2000)
    res = build_grid_meta_series(sig, window_seconds=1.0, rolling_history_seconds=5.0)
    for name in CHANNEL_NAMES:
        r = getattr(res, name)
        assert len(r.states) >= 2
        assert len(r.M_series) == len(r.states) - 1
        assert len(r.phases) == len(r.M_series)


def test_too_short_trace_raises():
    sig = _flat_signals(n=50)
    with pytest.raises(ValueError):
        build_grid_meta_series(sig, window_seconds=1.0, rolling_history_seconds=5.0)


# ---------------------------------------------------------------------
# Kontrola negatywna/pozytywna syntetyczna (male, szybkie slady)
# ---------------------------------------------------------------------

def test_negative_control_flat_noise_mostly_stable():
    """Czysty szum, zero zdarzen -> wiekszosc okien powinna wyjsc
    'stabilna' na kazdym kanale (patrz PROBA 1 w naglowku - to jest
    dokladnie test, ktory PROBA 1/WINDOW_SECONDS=0.5s NIE przechodzila).

    UWAGA (znalezione przy tym poprawianiu): pojedyncze ziarno na
    krotkim sladzie (n=4000 -> ~13 okien) to NIEDOMOCOWANY test tej
    hipotezy (patrz protokol numerologii, pkt. "sprawdz moc testu przed
    odczytaniem wyniku"). Realny, per-okienny wskaznik 'stabilna' przy
    WINDOW_SECONDS=2.0/ROLLING_HISTORY_SECONDS=10.0 to ok. 0.86-0.92
    (zmierzone empirycznie na 6 ziarnach x ~13 okien = 78 okien/kanal:
    voltage 0.923, frequency 0.885, harmonics 0.859, load 0.923). Na
    TAK MALEJ probie (13 okien) sam szum binarny potrafi dac pojedyncze
    ziarno w okolicy 0.69 (ziarno 42, kanal frequency: 9/13) - to nie
    byl blad adaptera, tylko oczekiwana wariancja malej proby. Naprawa:
    agregacja po kilku ziarnach (wiecej okien -> ciasniejszy szacunek
    prawdziwego wskaznika), zamiast dalszego podkrecania WINDOW_SECONDS
    dla jednego pechowego ziarna (to bylaby dokladnie ta sama pulapka
    post-hoc-tuningu, ktorej unikamy gdzie indziej w tym repo)."""
    seeds = (0, 1, 2, 3, 4, 42)
    totals = {name: [0, 0] for name in CHANNEL_NAMES}
    for seed in seeds:
        sig = _flat_signals(n=4000, seed=seed)
        res = build_grid_meta_series(sig, window_seconds=2.0, rolling_history_seconds=10.0)
        for name in CHANNEL_NAMES:
            r = getattr(res, name)
            totals[name][0] += sum(1 for p in r.phases if p == "stabilna")
            totals[name][1] += len(r.phases)

    for name, (n_stable, n_total) in totals.items():
        assert n_stable / n_total >= 0.75, (
            f"{name}: tylko {n_stable}/{n_total} 'stabilna' na czystym szumie "
            f"zagregowane po ziarnach {seeds} (oczekiwano >=0.75, zmierzona "
            f"realna wartosc bazowa to ok. 0.86-0.92)"
        )


def test_positive_control_voltage_dip_detected():
    """Krotki, gleboki zapad napiecia -> wiecej okien 'przejsciowa'/
    'krytyczna' na kanale voltage niz w kontroli negatywnej o tym samym
    ziarnie/dlugosci."""
    baseline = _flat_signals(n=4000, seed=7)
    dip = _voltage_dip_signals(n=4000, seed=7, dip_start=2000, dip_len=5)

    res_baseline = build_grid_meta_series(baseline, window_seconds=2.0, rolling_history_seconds=10.0)
    res_dip = build_grid_meta_series(dip, window_seconds=2.0, rolling_history_seconds=10.0)

    n_unstable_baseline = sum(1 for p in res_baseline.voltage.phases if p != "stabilna")
    n_unstable_dip = sum(1 for p in res_dip.voltage.phases if p != "stabilna")

    assert n_unstable_dip > n_unstable_baseline, (
        f"zapad napiecia nie podniosl liczby niestabilnych okien: "
        f"baseline={n_unstable_baseline}, dip={n_unstable_dip}"
    )


def test_dip_is_localized_to_window_containing_it():
    """Silniejszy test niz samo zliczanie: okno OBEJMUJACE zapad musi
    miec faze != 'stabilna' (albo krok M prowadzacy do/z niego)."""
    n, window_seconds, sample_rate = 4000, 2.0, 100.0
    dip_start = 2000  # t=20.0s
    sig = _voltage_dip_signals(n=n, seed=3, dip_start=dip_start, dip_len=5)
    res = build_grid_meta_series(sig, window_seconds=window_seconds, rolling_history_seconds=10.0)
    r = res.voltage

    dip_time = dip_start / sample_rate
    # znajdz indeks okna zawierajacego dip_time
    idx = None
    for i, ws in enumerate(r.window_starts):
        if ws <= dip_time < ws + window_seconds:
            idx = i
            break
    assert idx is not None, "test wewnetrzny: nie znaleziono okna z dip_time"

    # faza kroku M PROWADZACEGO DO tego okna (M_series[idx-1]) albo Z
    # niego (M_series[idx]) powinna byc != 'stabilna'
    nearby_phases = []
    if idx - 1 >= 0 and idx - 1 < len(r.phases):
        nearby_phases.append(r.phases[idx - 1])
    if idx < len(r.phases):
        nearby_phases.append(r.phases[idx])
    assert any(p != "stabilna" for p in nearby_phases), (
        f"zadna faza w oknach otaczajacych zapad (idx={idx}) nie jest niestabilna: {nearby_phases}"
    )


# ---------------------------------------------------------------------
# Scenariusze demo_generator.py (WOLNE - realne parametry na 60-80s sladach)
# ---------------------------------------------------------------------

@pytest.mark.slow
def test_real_scenario_normalny_is_mostly_stable_on_voltage():
    from demo_generator import generate
    sig = generate("normalny")
    res = build_grid_meta_series(sig)
    n_stable = sum(1 for p in res.voltage.phases if p == "stabilna")
    assert n_stable / len(res.voltage.phases) >= 0.9


@pytest.mark.slow
def test_real_scenario_mikrozanik_elevates_voltage_instability_vs_normalny():
    from demo_generator import generate
    normalny = build_grid_meta_series(generate("normalny"))
    mikrozanik = build_grid_meta_series(generate("mikrozanik"))

    n_unstable_normalny = sum(1 for p in normalny.voltage.phases if p != "stabilna")
    n_unstable_mikrozanik = sum(1 for p in mikrozanik.voltage.phases if p != "stabilna")

    assert n_unstable_mikrozanik > n_unstable_normalny


@pytest.mark.slow
def test_real_scenario_cykliczne_zaklocenia_not_detected_on_load():
    """Udokumentowany, PRZEWIDZIANY PRZED URUCHOMIENIEM wynik negatywny
    (zastrzezenie #3 w naglowku modulu): M-operator nie wykrywa stanu
    okresowego, tylko zmiane. Ten test PILNUJE tego zalozenia - jesli
    kiedys zacznie failowac (bo ktos zmieni okna/progi), to znak, ze
    trzeba zaktualizowac dokumentacje, nie ze test jest zly."""
    from demo_generator import generate
    res = build_grid_meta_series(generate("cykliczne_zaklocenia"))
    n_unstable = sum(1 for p in res.load.phases if p != "stabilna")
    assert n_unstable == 0, (
        "cykliczne_zaklocenia zostalo wykryte na load - zaktualizuj "
        "dokumentacje w naglowku meta_adapter.py (zastrzezenie #3), bo "
        "zalozenie architektoniczne przestalo byc prawdziwe"
    )
