"""
forecast_core.py — prognoza kolejnego odczytu per kanał (napięcie/
częstotliwość/obciążenie/harmoniczne) + interpretacja zdarzeń
prognozowanych (przewidywane przeciążenie/mikro-zanik/skok THD).

UWAGA nt. wyboru modelu — dlaczego NIE LSTM/PyTorch:
Dostarczony szkic użytkownika używał sieci LSTM (PyTorch). W środowisku,
w którym budowany jest ten moduł, instalacja `torch` zakończyła się
błędem braku miejsca na dysku (pakiet ma kilka GB). Niezależnie od tego:
sieć z LOSOWO zainicjalizowanymi wagami (bo `model_path` jest opcjonalne,
a żadnego wytrenowanego pliku wag nie dostarczono) nie daje sensownej
prognozy - dałaby TYLKO fikcyjne liczby, wyglądające jak działający model
AI, ale bez żadnej wartości predykcyjnej. To ten sam problem co przy
`device_client.py` (nigdy nie udawaj działania - jawny brak jest lepszy
niż fałszywe dane).

Zamiast tego: domyślny, w pełni przetestowany model to **eksponencjalne
wygładzanie z trendem (metoda Holta)** per kanał - zero dodatkowych
zależności (tylko numpy, już wymagany), deterministyczny, łatwy do
zweryfikowania (stały sygnał → stała prognoza, trend liniowy →
poprawna ekstrapolacja nachylenia). To NIE jest "gorsza wersja LSTM" -
to świadomy wybór: prosty, przejrzysty model statystyczny, który
faktycznie robi to, co obiecuje, zamiast nieprzetestowanej sieci
neuronowej udającej inteligencję bez treningu na realnych danych.

Jeśli w przyszłości pojawi się realny, wytrenowany na rzeczywistych
odczytach model (LSTM albo inny), można go podłączyć zamiast
`HoltTrendPredictor` - `TimdrEnergyPredictor.predict_next()` to jedyny
kontrakt, którego reszta modułu (`TimdrEnergyForecaster`) wymaga.
"""

from __future__ import annotations

import numpy as np

CHANNELS = ("voltage", "frequency", "load", "harmonic")


class HoltTrendPredictor:
    """Eksponencjalne wygładzanie z trendem (Holt) - lekki model
    autoregresyjny jednego kanału, bez zależności poza numpy.

    alpha - waga nowego poziomu (0,1]; beta - waga nowego trendu [0,1].
    Wyższe wartości = szybsza reakcja na zmiany, niższe = więcej
    wygładzenia/mniej szumu.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        if not (0 < alpha <= 1):
            raise ValueError("HoltTrendPredictor: alpha musi być w przedziale (0, 1].")
        if not (0 <= beta <= 1):
            raise ValueError("HoltTrendPredictor: beta musi być w przedziale [0, 1].")
        self.alpha = alpha
        self.beta = beta

    def predict_next(self, series) -> float:
        series = np.asarray(series, dtype=float)
        if len(series) == 0:
            raise ValueError("HoltTrendPredictor.predict_next: pusta seria - brak danych do prognozy.")
        if len(series) == 1:
            return float(series[0])

        level = series[0]
        trend = series[1] - series[0]
        for x in series[1:]:
            prev_level = level
            level = self.alpha * x + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
        return float(level + trend)


class TimdrEnergyPredictor:
    """Prognoza JEDNEGO kroku naprzód, niezależnie per kanał, z ostatniego
    okna historii. Patrz docstring modułu odnośnie wyboru modelu."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self._predictors = {ch: HoltTrendPredictor(alpha=alpha, beta=beta) for ch in CHANNELS}

    def predict_next(self, voltage, frequency, load, harmonic) -> dict:
        return {
            "voltage_next": self._predictors["voltage"].predict_next(voltage),
            "frequency_next": self._predictors["frequency"].predict_next(frequency),
            "load_next": self._predictors["load"].predict_next(load),
            "harmonic_next": self._predictors["harmonic"].predict_next(harmonic),
        }


class TimdrEnergyForecastEvents:
    def __init__(self):
        self.predicted_overload = None       # dict albo None
        self.predicted_micro_outage = None    # dict albo None
        self.predicted_harmonic_spike = None  # dict albo None

    def any(self) -> bool:
        return any((self.predicted_overload, self.predicted_micro_outage, self.predicted_harmonic_spike))


class TimdrEnergyForecaster:
    """
    Interpretacja prognozy `TimdrEnergyPredictor.predict_next()` jako
    zdarzeń ("za N próbek ryzyko przeciążenia" itd).

    Naprawione względem dostarczonego szkicu (2 błędy - patrz
    README.md, sekcja "Uwagi techniczne"):

    1. Przeciążenie porównywało `load_next > overload_threshold *
       load_next` - czyli WARTOŚĆ DO SAMEJ SIEBIE pomnożonej przez 0.9.
       Dla dowolnej dodatniej wartości `x > 0.9*x` jest ZAWSZE prawdziwe
       - to była gwarantowana fałszywa flaga na KAŻDEJ prognozie z
       dodatnim obciążeniem. Naprawione: porównanie do
       `overload_threshold * rated_load` (tak jak w `grid_monitor.py`) -
       wymaga jawnego podania `rated_load` w konstruktorze, bez cichego
       fallbacku.
    2. Analogiczny błąd dla THD: `harmonic_next > harmonic_spike_factor *
       harmonic_next` (x > 3*x) jest prawdziwe TYLKO dla x < 0 - dla
       THD (zawsze ≥ 0) ten warunek nigdy się nie uruchamiał, więc
       detektor skoków harmonicznych był martwym kodem. Naprawione:
       podwójny próg jak w `grid_monitor._detect_harmonic_anomalies` -
       stały limit normy EN 50160 (8%) ORAZ opcjonalny próg adaptacyjny
       (MAD-z) liczony z `recent_harmonics`, jeśli podane.
    """

    def __init__(
        self,
        rated_load: float,
        v_nominal: float = 230.0,
        f_nominal: float = 50.0,
        overload_threshold: float = 0.9,
        micro_outage_drop: float = 0.5,
        harmonic_thd_limit_pct: float = 8.0,
        harmonic_adaptive_factor: float = 3.0,
    ):
        if rated_load is None or rated_load <= 0:
            raise ValueError(
                "TimdrEnergyForecaster wymaga podania dodatniego 'rated_load' (moc "
                "znamionowa w W) - próg przewidywanego przeciążenia liczony jest "
                "WZGLĘDEM NIEJ, nigdy względem samej prognozowanej wartości."
            )
        self.rated_load = rated_load
        self.v_nominal = v_nominal
        self.f_nominal = f_nominal
        self.overload_threshold = overload_threshold
        self.micro_outage_drop = micro_outage_drop
        self.harmonic_thd_limit_pct = harmonic_thd_limit_pct
        self.harmonic_adaptive_factor = harmonic_adaptive_factor

    def analyze_prediction(self, prediction: dict, recent_harmonics=None) -> TimdrEnergyForecastEvents:
        events = TimdrEnergyForecastEvents()

        threshold_load = self.overload_threshold * self.rated_load
        if prediction["load_next"] > threshold_load:
            events.predicted_overload = {
                "load_next": prediction["load_next"],
                "threshold": threshold_load,
                "rated_load": self.rated_load,
            }

        threshold_v = self.micro_outage_drop * self.v_nominal
        if prediction["voltage_next"] < threshold_v:
            events.predicted_micro_outage = {
                "voltage_next": prediction["voltage_next"],
                "threshold": threshold_v,
            }

        thd = prediction["harmonic_next"]
        over_limit = thd > self.harmonic_thd_limit_pct
        over_adaptive = False
        if recent_harmonics is not None:
            arr = np.asarray(recent_harmonics, dtype=float)
            if len(arr) >= 8:
                med = np.median(arr)
                mad = np.median(np.abs(arr - med))
                spread = 1.4826 * mad if mad > 0 else (np.ptp(arr) / 4.0 or 1e-6)
                z = (thd - med) / spread
                over_adaptive = bool(z > self.harmonic_adaptive_factor)

        if over_limit or over_adaptive:
            if over_limit and over_adaptive:
                powod = "oba"
            elif over_limit:
                powod = "limit_normy"
            else:
                powod = "odchylenie_adaptacyjne"
            events.predicted_harmonic_spike = {"harmonic_next": thd, "powod": powod}

        return events
