"""Adapter PROTECT-90 to TimdrEnergySignals.

PROTECT-90 contains raw three-phase EMT waveforms, not ready-made
voltage/frequency/THD/load logs. This module makes every reduction explicit.
It is for research benchmarking, not a certified protection function.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grid_monitor import TimdrEnergySignals

PROTECT90_SAMPLE_RATE_HZ = 6400.0
OUTPUT_RATE_HZ = 100.0
SAMPLES_PER_OUTPUT = int(PROTECT90_SAMPLE_RATE_HZ / OUTPUT_RATE_HZ)


class Protect90Error(ValueError):
    """Raised when a PROTECT-90 file does not match its documented schema."""


def _episode_path(data_dir: str | Path, sample_id: int) -> Path:
    root = Path(data_dir)
    name = f"{int(sample_id)}_sample_hv_double_line_90kv.pkl"
    candidates = (
        root / "preprocessed_data" / name,
        root / "hv_double_line_90kv_preprocessed_data" / name,
        root / name,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Nie znaleziono epizodu " + str(sample_id))


def _locations(columns: list[str]) -> list[str]:
    suffix = "_vol_L1_V"
    return sorted({column[:-len(suffix)] for column in columns if column.endswith(suffix)})


def _rms_blocks(values: np.ndarray) -> np.ndarray:
    n = len(values) // SAMPLES_PER_OUTPUT * SAMPLES_PER_OUTPUT
    if n == 0:
        raise Protect90Error("Za mało próbek do utworzenia bloku 10 ms.")
    blocks = values[:n].reshape(-1, SAMPLES_PER_OUTPUT)
    return np.sqrt(np.mean(blocks ** 2, axis=1))


def _thd_blocks(values: np.ndarray) -> np.ndarray:
    """THD-like ratio from a one-cycle Hann-windowed FFT per output block."""
    n_blocks = len(values) // SAMPLES_PER_OUTPUT
    cycle = int(PROTECT90_SAMPLE_RATE_HZ / 50.0)
    window = np.hanning(cycle)
    result = []
    for block in range(n_blocks):
        center = block * SAMPLES_PER_OUTPUT + SAMPLES_PER_OUTPUT // 2
        start = min(max(center - cycle // 2, 0), len(values) - cycle)
        spectrum = np.abs(np.fft.rfft(values[start:start + cycle] * window))
        fundamental = spectrum[1] if len(spectrum) > 1 else 0.0
        ratio = np.sqrt(float(np.sum(spectrum[2:] ** 2))) / fundamental if fundamental > 1e-12 else 0.0
        result.append(100.0 * ratio)
    return np.asarray(result, dtype=float)


def _frequency_blocks(values: np.ndarray) -> np.ndarray:
    """Estimate frequency from upward zero crossings over a 50 ms window."""
    n_blocks = len(values) // SAMPLES_PER_OUTPUT
    width = int(PROTECT90_SAMPLE_RATE_HZ * 0.05)
    result = []
    for block in range(n_blocks):
        center = block * SAMPLES_PER_OUTPUT + SAMPLES_PER_OUTPUT // 2
        start = min(max(center - width // 2, 0), len(values) - width)
        segment = values[start:start + width]
        crossings = np.flatnonzero((segment[:-1] <= 0) & (segment[1:] > 0))
        if len(crossings) >= 2:
            left = segment[crossings]
            right = segment[crossings + 1]
            fractional = crossings - left / (right - left)
            result.append(float(1.0 / np.median(np.diff(fractional) / PROTECT90_SAMPLE_RATE_HZ)))
        else:
            result.append(50.0)
    return np.asarray(result, dtype=float)


def load_protect90_episode(data_dir: str | Path, sample_id: int, location: str | None = None) -> tuple[TimdrEnergySignals, float, dict[str, Any]]:
    """Load one episode and derive four aligned Grid Monitor channels."""
    path = _episode_path(data_dir, sample_id)
    frame = pd.read_pickle(path)
    if "time_s" not in frame.columns:
        raise Protect90Error("Brak wymaganej kolumny time_s.")
    available = _locations(list(frame.columns))
    if not available:
        raise Protect90Error("Nie znaleziono trójfazowych kanałów napięciowych PROTECT-90.")
    chosen = location or available[0]
    expected = [f"{chosen}_{kind}_L{phase}_{unit}" for kind, unit in (("vol", "V"), ("cur", "A")) for phase in (1, 2, 3)]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise Protect90Error(f"Lokalizacja {chosen!r} nie ma kanałów: {missing}")
    va, vb, vc = (frame[f"{chosen}_vol_L{phase}_V"].to_numpy(float) for phase in (1, 2, 3))
    ia, ib, ic = (frame[f"{chosen}_cur_L{phase}_A"].to_numpy(float) for phase in (1, 2, 3))
    voltage = np.mean(np.vstack([_rms_blocks(va), _rms_blocks(vb), _rms_blocks(vc)]), axis=0)
    current = np.mean(np.vstack([_rms_blocks(ia), _rms_blocks(ib), _rms_blocks(ic)]), axis=0)
    signals = TimdrEnergySignals(voltage, _frequency_blocks(va), _thd_blocks(va), np.sqrt(3.0) * voltage * current)
    provenance = {
        "dataset": "PROTECT-90 v1.0.0", "sample_id": int(sample_id),
        "source_file": str(path), "location": chosen, "available_locations": available,
        "sample_rate_hz": OUTPUT_RATE_HZ,
        "voltage": "mean three-phase 10 ms RMS",
        "frequency": "upward-zero-crossing estimate; 50 Hz fallback when crossings absent",
        "harmonics": "one-cycle Hann-windowed FFT ratio, percent",
        "load": "apparent-power proxy sqrt(3)*Vrms*Irms; not measured active power",
    }
    return signals, OUTPUT_RATE_HZ, provenance
