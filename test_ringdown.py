"""Testy ringdown.py - port 1:1 z universal-state-analyzer/timdr_core/ringdown.py
(patrz tam po pełną walidację numeryczną: odzyskana częstotliwość/tłumienie
zgodne z teorią na tłumionym oscylatorze o znanym f0/tau). Tu: powtórzona
kluczowa walidacja + wariant specyficzny dla sieci energetycznej
(baseline=f_nominal jawnie, nie średnia z okna przed zdarzeniem).
"""
import numpy as np
import pytest

from ringdown import ringdown_resonance


def test_underdamped_recovers_known_frequency_and_damping():
    fs = 1000.0  # próbki/s - realistyczne dla monitoringu częstotliwości sieci
    t = np.arange(0, 8.0, 1 / fs)
    event_idx = int(2.0 * fs)
    f0, tau = 0.5, 1.5  # typowe dla oscylacji mocy w sieci: rząd 0.1-2 Hz
    post = t[event_idx:] - t[event_idx]
    x = np.zeros_like(t)
    x[event_idx:] = 5.0 * np.exp(-post / tau) * np.cos(2 * np.pi * f0 * post)
    rng = np.random.default_rng(0)
    x_noisy = x + rng.normal(0, 0.05, len(t))

    res = ringdown_resonance(t, x_noisy, event_idx=event_idx, pre_event_window=event_idx)
    assert res["is_oscillatory"] is True
    assert res["frequency_hz"] == pytest.approx(f0, rel=0.05)
    zeta_theory = 1.0 / np.sqrt((2 * np.pi * f0 * tau) ** 2 + 1)
    assert res["damping_ratio"] == pytest.approx(zeta_theory, rel=0.3)


def test_monotonic_recovery_is_not_oscillatory():
    fs = 1000.0
    t = np.arange(0, 8.0, 1 / fs)
    event_idx = int(2.0 * fs)
    post = t[event_idx:] - t[event_idx]
    x = np.zeros_like(t)
    x[event_idx:] = 5.0 * np.exp(-post / 1.0)
    rng = np.random.default_rng(1)
    x_noisy = x + rng.normal(0, 0.05, len(t))

    res = ringdown_resonance(t, x_noisy, event_idx=event_idx, pre_event_window=event_idx)
    assert res["is_oscillatory"] is False


def test_explicit_baseline_f_nominal_used_instead_of_pre_event_mean():
    """Wzorzec z grid_monitor.py: baseline=f_nominal PODANE JAWNIE (nie
    liczone z lokalnej historii) - sieć ma realny, znany z góry punkt
    odniesienia."""
    fs = 1000.0
    t = np.arange(0, 4.0, 1 / fs)
    event_idx = int(1.0 * fs)
    post = t[event_idx:] - t[event_idx]
    freq = np.full_like(t, 51.0)  # drift, NIE 50.0 - baseline musi zignorować to i wziąć f_nominal
    freq[event_idx:] = 50.0 + 2.0 * np.exp(-post / 0.5) * np.cos(2 * np.pi * 1.0 * post)

    res = ringdown_resonance(t, freq, event_idx=event_idx, baseline=50.0, pre_event_window=event_idx)
    assert res["baseline"] == 50.0
