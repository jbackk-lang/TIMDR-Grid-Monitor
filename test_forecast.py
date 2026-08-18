"""test_forecast.py — testy dla forecast_core.py, w tym regresje 2
błędów znalezionych w dostarczonym szkicu (TimdrEnergyForecaster
porównywał prognozowaną wartość do samej siebie zamiast do progu)."""

import numpy as np
import pytest

from forecast_core import (
    HoltTrendPredictor, TimdrEnergyPredictor,
    TimdrEnergyForecastEvents, TimdrEnergyForecaster,
)


# ---------------------------------------------------------------------
# HoltTrendPredictor
# ---------------------------------------------------------------------

def test_holt_odrzuca_niepoprawna_alpha():
    with pytest.raises(ValueError, match="alpha"):
        HoltTrendPredictor(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        HoltTrendPredictor(alpha=1.5)


def test_holt_odrzuca_niepoprawna_beta():
    with pytest.raises(ValueError, match="beta"):
        HoltTrendPredictor(beta=-0.1)


def test_holt_pusta_seria_rzuca_blad():
    p = HoltTrendPredictor()
    with pytest.raises(ValueError, match="pusta"):
        p.predict_next([])


def test_holt_jedna_wartosc_zwraca_ja_samą():
    p = HoltTrendPredictor()
    assert p.predict_next([42.0]) == pytest.approx(42.0)


def test_holt_stala_seria_daje_stala_prognoze():
    p = HoltTrendPredictor(alpha=0.5, beta=0.2)
    series = np.full(50, 230.0)
    pred = p.predict_next(series)
    assert pred == pytest.approx(230.0, abs=0.01)


def test_holt_trend_liniowy_ekstrapoluje_nachylenie():
    p = HoltTrendPredictor(alpha=0.8, beta=0.8)
    series = np.arange(0, 100, 1.0)  # 0,1,2,...,99 - nachylenie 1/krok
    pred = p.predict_next(series)
    # kolejna wartość "powinna" być ~100, dopuszczamy pewien błąd wygładzania
    assert 95 <= pred <= 105


# ---------------------------------------------------------------------
# TimdrEnergyPredictor
# ---------------------------------------------------------------------

def test_predictor_zwraca_cztery_klucze():
    pred = TimdrEnergyPredictor()
    rng = np.random.default_rng(0)
    n = 30
    out = pred.predict_next(
        voltage=230.0 + rng.normal(0, 0.3, n),
        frequency=50.0 + rng.normal(0, 0.02, n),
        load=2000.0 + rng.normal(0, 50, n),
        harmonic=np.abs(rng.normal(2.0, 0.3, n)),
    )
    assert set(out.keys()) == {"voltage_next", "frequency_next", "load_next", "harmonic_next"}
    assert all(isinstance(v, float) for v in out.values())


# ---------------------------------------------------------------------
# TimdrEnergyForecaster - walidacja
# ---------------------------------------------------------------------

def test_forecaster_wymaga_dodatniego_rated_load():
    with pytest.raises(ValueError, match="rated_load"):
        TimdrEnergyForecaster(rated_load=None)
    with pytest.raises(ValueError, match="rated_load"):
        TimdrEnergyForecaster(rated_load=0)
    with pytest.raises(ValueError, match="rated_load"):
        TimdrEnergyForecaster(rated_load=-100)


# ---------------------------------------------------------------------
# Regresja bug 1: przeciążenie liczone względem rated_load, nie samo od siebie
# ---------------------------------------------------------------------

def test_regresja_overload_nie_flaguje_zawsze():
    """W szkicu: `load_next > 0.9 * load_next` jest PRAWDĄ dla KAŻDEJ
    dodatniej wartości - gwarantowany fałszywy alarm na każdej prognozie.
    Tu: przy rated_load wystarczająco dużym, żadnego alarmu."""
    forecaster = TimdrEnergyForecaster(rated_load=100_000.0)
    prediction = {"voltage_next": 230.0, "frequency_next": 50.0, "load_next": 5000.0, "harmonic_next": 2.0}
    events = forecaster.analyze_prediction(prediction)
    assert events.predicted_overload is None


def test_regresja_overload_wykrywa_realne_przeciazenie_wzgledem_rated_load():
    forecaster = TimdrEnergyForecaster(rated_load=10_000.0, overload_threshold=0.9)
    prediction = {"voltage_next": 230.0, "frequency_next": 50.0, "load_next": 9500.0, "harmonic_next": 2.0}
    events = forecaster.analyze_prediction(prediction)
    assert events.predicted_overload is not None
    assert events.predicted_overload["threshold"] == pytest.approx(9000.0)


def test_overload_ten_sam_load_next_flaguje_lub_nie_zaleznie_od_rated_load():
    """Kluczowy dowód naprawy: DOKŁADNIE ta sama prognozowana wartość
    load_next=5000 raz flaguje, raz nie - zależnie od rated_load, a NIE
    zawsze (jak w oryginalnym buggy self-comparison)."""
    prediction = {"voltage_next": 230.0, "frequency_next": 50.0, "load_next": 5000.0, "harmonic_next": 2.0}

    low_rated = TimdrEnergyForecaster(rated_load=4000.0, overload_threshold=0.9)
    assert low_rated.analyze_prediction(prediction).predicted_overload is not None

    high_rated = TimdrEnergyForecaster(rated_load=100_000.0, overload_threshold=0.9)
    assert high_rated.analyze_prediction(prediction).predicted_overload is None


# ---------------------------------------------------------------------
# Regresja bug 2: skok harmonicznych liczony względem realnego progu
# ---------------------------------------------------------------------

def test_regresja_harmonic_spike_nie_jest_martwym_kodem():
    """W szkicu: `harmonic_next > 3 * harmonic_next` jest prawdą TYLKO
    dla ujemnych wartości - dla THD (zawsze >= 0) nigdy się nie
    uruchamiało. Tu: realne przekroczenie normy 8% faktycznie flaguje."""
    forecaster = TimdrEnergyForecaster(rated_load=10_000.0)
    prediction = {"voltage_next": 230.0, "frequency_next": 50.0, "load_next": 2000.0, "harmonic_next": 9.5}
    events = forecaster.analyze_prediction(prediction)
    assert events.predicted_harmonic_spike is not None
    assert events.predicted_harmonic_spike["powod"] in ("limit_normy", "oba")


def test_harmonic_ponizej_normy_bez_historii_nie_flaguje():
    forecaster = TimdrEnergyForecaster(rated_load=10_000.0)
    prediction = {"voltage_next": 230.0, "frequency_next": 50.0, "load_next": 2000.0, "harmonic_next": 2.0}
    events = forecaster.analyze_prediction(prediction)
    assert events.predicted_harmonic_spike is None


def test_harmonic_adaptacyjny_z_historia_flaguje_odchylenie_ponizej_normy():
    rng = np.random.default_rng(1)
    recent = np.abs(rng.normal(2.0, 0.2, 50))  # baseline nisko, daleko od 8%
    forecaster = TimdrEnergyForecaster(rated_load=10_000.0, harmonic_adaptive_factor=3.0)
    prediction = {"voltage_next": 230.0, "frequency_next": 50.0, "load_next": 2000.0, "harmonic_next": 4.0}
    events = forecaster.analyze_prediction(prediction, recent_harmonics=recent)
    assert events.predicted_harmonic_spike is not None
    assert events.predicted_harmonic_spike["powod"] == "odchylenie_adaptacyjne"


# ---------------------------------------------------------------------
# Mikro-zanik (ten fragment szkicu był poprawny - test sanity)
# ---------------------------------------------------------------------

def test_micro_outage_wykrywa_przewidywany_spadek_napiecia():
    forecaster = TimdrEnergyForecaster(rated_load=10_000.0, v_nominal=230.0, micro_outage_drop=0.5)
    prediction = {"voltage_next": 90.0, "frequency_next": 50.0, "load_next": 2000.0, "harmonic_next": 2.0}
    events = forecaster.analyze_prediction(prediction)
    assert events.predicted_micro_outage is not None


def test_micro_outage_zdrowe_napiecie_nie_flaguje():
    forecaster = TimdrEnergyForecaster(rated_load=10_000.0)
    prediction = {"voltage_next": 229.0, "frequency_next": 50.0, "load_next": 2000.0, "harmonic_next": 2.0}
    events = forecaster.analyze_prediction(prediction)
    assert events.predicted_micro_outage is None


def test_events_any_helper():
    events = TimdrEnergyForecastEvents()
    assert events.any() is False
    events.predicted_micro_outage = {"x": 1}
    assert events.any() is True
