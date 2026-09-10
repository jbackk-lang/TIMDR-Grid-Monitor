"""_vendor_seismic_meta_windowing.py -- ZWENDOROWANA (skopiowana 1:1, NIE
sibling-importowana) kopia generycznej warstwy okienkowania z
TIMDR-Earthquake-Core/meta_adapter.py: SeismicMetaResult,
compute_global_thresholds, window_to_meta_state, build_meta_series_from_waveform.

POCHODZENIE I DLACZEGO ZWENDOROWANE: ta warstwa NIE jest w swojej logice
specyficzna dla sejsmiki -- działa na dowolnej parze (t,s) i przyjmuje
`core` (obiekt z metodami flow()/trm()) jako parametr, dlatego
meta_adapter.py w tym repo (TIMDR-Grid-Monitor) reużywał jej wprost przez
sys.path sibling-import z folderu-siostry TIMDR-Earthquake-Core, zamiast
przepisywać tę samą matematykę okienkowania/progowania po raz kolejny.
Zwendorowano dnia 2026-09-10, żeby to repo było samowystarczalne po
sklonowaniu SAMEGO SIEBIE (decyzja podjęta na wyraźną prośbę:
"repozytoria kodu mają być niezależne od siebie"). To jest KOPIA, nie
link -- pełne uzasadnienie wzorów/progów/zastrzeżenia (PRE-REJESTRACJA V2,
PROBA 1/2, zastrzeżenia #1-#7) są udokumentowane w źródle,
`TIMDR-Earthquake-Core/meta_adapter.py` -- tu przeniesiony jest tylko
działający kod, bez powielania całego tamtego eseju; jeśli te wzory mają
być kiedyś zmienione, zrób to świadomie w OBU miejscach (tu i w źródle)
albo zaakceptuj rozjazd.

Klasa `SeismicMetaResult`/nazwy funkcji zachowane bez zmian (przenoszenie
nazwy byłoby czysto kosmetyczne, patrz oryginalny docstring
TIMDR-Grid-Monitor/meta_adapter.py REUZYCIE).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from _vendor_timdr_meta_dynamics_core import MetaState, MetaOperatorM, MetaMap, MetaTrigger, MetaTriggerResult
from _vendor_timdr_core_earthquake import TIMDR_EarthquakeCore

MAD_TO_STD = 1.4826  # ta sama stala co Synoptyk-v3/membrane/defects.py
ROBUST_K = 3.5        # ta sama wartosc co Synoptyk-v3 DEFECT_K
ANOMALY_FACTOR = 3.5  # jw., przekazywane do core.anomalies()

WINDOW_SECONDS_DEFAULT = 5.0


@dataclass
class SeismicMetaResult:
    window_starts: List[float]     # czas poczatku kazdego okna [s], dlugosc n
    states: List[MetaState]        # S_meta(t) per okno, dlugosc n
    M_series: List[MetaState]      # M(t) = dS/dt, dlugosc n-1
    phases: List[str]              # faza per krok M, dlugosc n-1
    trigger: MetaTriggerResult


def _robust_threshold(values: np.ndarray, k: float = ROBUST_K) -> float:
    """Mediana + k*MAD(przeskalowany)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med))) * MAD_TO_STD
    return med + k * mad


def _high_freq_fraction(s_window: np.ndarray) -> float:
    """Ulamek energii widmowej w gornej polowie pasm czestotliwosci."""
    n = len(s_window)
    if n < 4:
        return 0.0
    centered = s_window - np.mean(s_window)
    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2
    half = len(power) // 2
    low = float(power[:half].sum())
    high = float(power[half:].sum())
    total = low + high
    return high / total if total > 0 else 0.0


@dataclass
class GlobalThresholds:
    """Progi policzone RAZ na CALYM sladzie (nie per-okno) -- patrz
    uzasadnienie (normalizacja mean(okno)/prog(TEGO SAMEGO okna) jest
    algebraicznie niezmiennicza na jednorodne przeskalowanie okna) w
    zrodle, TIMDR-Earthquake-Core/meta_adapter.py."""
    flow_threshold: float
    twist_threshold: float
    anomaly_threshold: float


def compute_global_thresholds(core: TIMDR_EarthquakeCore, t: np.ndarray, s: np.ndarray) -> GlobalThresholds:
    """Liczy progi robust na PODANYM (t,s) -- PRZED podzieleniem na okna."""
    flow_grad_all = core.flow(t, s)
    flow_threshold = _robust_threshold(np.abs(flow_grad_all))

    twist_strength_all = np.abs(np.gradient(flow_grad_all, t))
    twist_threshold = _robust_threshold(twist_strength_all)

    smooth_all = core.trm(t, s)
    residuals_all = s - smooth_all
    mad = float(np.median(np.abs(residuals_all))) * MAD_TO_STD
    if mad <= 1e-12:
        std = float(np.std(residuals_all))
        mad = std if std > 1e-12 else 1e-9
    anomaly_threshold = ANOMALY_FACTOR * mad

    return GlobalThresholds(
        flow_threshold=flow_threshold,
        twist_threshold=twist_threshold,
        anomaly_threshold=anomaly_threshold,
    )


def window_to_meta_state(
    core: TIMDR_EarthquakeCore,
    t_window: np.ndarray,
    s_window: np.ndarray,
    thresholds: GlobalThresholds,
) -> MetaState:
    """Mapowanie jednego okna (t,s) -> jeden MetaState, wzgledem progow
    GLOBALNYCH (`thresholds`, policzonych raz na calym sladzie)."""
    n = len(s_window)

    Lambda = _high_freq_fraction(s_window)

    flow_grad = core.flow(t_window, s_window)
    abs_flow = np.abs(flow_grad)
    tau = float(abs_flow.mean()) / thresholds.flow_threshold if thresholds.flow_threshold > 0 else 0.0

    if n >= 3:
        twist_strength = np.abs(np.gradient(flow_grad, t_window))
        n_twist = int(np.sum(twist_strength > thresholds.twist_threshold))
    else:
        n_twist = 0
    J = n_twist / n if n > 0 else 0.0

    smooth = core.trm(t_window, s_window)
    residuals = s_window - smooth
    n_anomaly = int(np.sum(np.abs(residuals) > thresholds.anomaly_threshold))
    rho = n_anomaly / n if n > 0 else 0.0

    return MetaState(Lambda=Lambda, tau=tau, rho=rho, J=J)


def build_meta_series_from_waveform(
    t: np.ndarray,
    s: np.ndarray,
    window_seconds: float = WINDOW_SECONDS_DEFAULT,
    core: Optional[TIMDR_EarthquakeCore] = None,
    dt: Optional[float] = None,
    calibration_end: Optional[float] = None,
    rolling_history_seconds: Optional[float] = None,
) -> SeismicMetaResult:
    """Dzieli (t,s) na kolejne, NIENAKLADAJACE SIE okna dlugosci
    `window_seconds`, liczy MetaState per okno, potem M-serie/fazy/trigger.

    Trzy WZAJEMNIE WYKLUCZAJACE SIE tryby kalibracji progow -- pelny opis
    (i uzasadnienie kazdego) w zrodle, TIMDR-Earthquake-Core/meta_adapter.py:
    (1) domyslny: progi z CALEGO sladu (lookahead, ale pelna struktura);
    (2) `calibration_end=<czas>`: progi tylko sprzed tego czasu, stale dalej;
    (3) `rolling_history_seconds=<s>`: progi przeliczane per okno z trailing
        historii -- ZALECANY wariant do ciaglej detekcji (patrz zrodlo)."""
    t = np.asarray(t, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if len(t) != len(s):
        raise ValueError(f"t i s musza miec ta sama dlugosc, dostano {len(t)} i {len(s)}")
    if len(t) < 2:
        raise ValueError("Potrzeba >= 2 probek")
    if calibration_end is not None and rolling_history_seconds is not None:
        raise ValueError(
            "calibration_end i rolling_history_seconds sa wzajemnie "
            "wykluczajace - podaj co najwyzej jedno."
        )

    if core is None:
        core = TIMDR_EarthquakeCore()
    if dt is None:
        dt = window_seconds

    t0 = t[0]
    duration = t[-1] - t0
    n_windows = int(duration // window_seconds)
    if n_windows < 2:
        raise ValueError(
            f"Za krotki slad ({duration:.1f}s) na >= 2 pelne okna po "
            f"{window_seconds}s - dostano {n_windows}."
        )

    window_starts: List[float] = []
    states: List[MetaState] = []

    if rolling_history_seconds is None:
        if calibration_end is None:
            thresholds = compute_global_thresholds(core, t, s)
        else:
            calib_mask = t < calibration_end
            if calib_mask.sum() < 4:
                raise ValueError(
                    f"calibration_end={calibration_end} zostawia tylko "
                    f"{int(calib_mask.sum())} probek referencyjnych (< 4)."
                )
            thresholds = compute_global_thresholds(core, t[calib_mask], s[calib_mask])

        for i in range(n_windows):
            w_start = t0 + i * window_seconds
            w_end = w_start + window_seconds
            mask = (t >= w_start) & (t < w_end)
            t_win, s_win = t[mask], s[mask]
            if len(t_win) < 4:
                continue
            window_starts.append(w_start)
            states.append(window_to_meta_state(core, t_win, s_win, thresholds))
    else:
        for i in range(n_windows):
            w_start = t0 + i * window_seconds
            w_end = w_start + window_seconds
            hist_start = w_start - rolling_history_seconds
            if hist_start < t0:
                continue
            hist_mask = (t >= hist_start) & (t < w_start)
            if hist_mask.sum() < 4:
                continue
            window_mask = (t >= w_start) & (t < w_end)
            t_win, s_win = t[window_mask], s[window_mask]
            if len(t_win) < 4:
                continue
            thresholds_i = compute_global_thresholds(core, t[hist_mask], s[hist_mask])
            window_starts.append(w_start)
            states.append(window_to_meta_state(core, t_win, s_win, thresholds_i))

    if len(states) < 2:
        raise ValueError(f"Za malo pelnych okien z wystarczajaca liczba probek ({len(states)} < 2)")

    meta_operator = MetaOperatorM()
    M_series: List[MetaState] = []
    for i in range(len(states) - 1):
        M_series.append(meta_operator.compute(states[i], states[i + 1], dt))

    meta_map = MetaMap(meta_operator)
    phases = meta_map.detect_transitions(M_series)
    trigger = MetaTrigger().analyze(phases)

    return SeismicMetaResult(
        window_starts=window_starts, states=states,
        M_series=M_series, phases=phases, trigger=trigger,
    )
