"""Smoke test dla real_protect90_membrane_bridge.py.

NIE testuje na prawdziwych 512 epizodach (za wolne/zbyt duze dla testu
jednostkowego) -- buduje maly, syntetyczny zestaw plikow o DOKLADNIE
takim samym uklada jak prawdziwy zbior PROTECT-90 (ten sam uklad
katalogow, te same nazwy kolumn/plikow), podmienia sciezki modulu na
katalog tymczasowy (monkeypatch), i sprawdza, ze caly pipeline
(wczytanie -> ekstrakcja okien baseline/event -> test Wilcoxona
sparowany -> klasyfikacja) przechodzi bez wywalenia i daje sensowny
ksztalt wyniku. Sprawdza rowniez wprost poprawnosc znaku
`paired_wilcoxon_test` (blad znaku `scipy.stats.wilcoxon` odkryty i
naprawiony w tej sesji, patrz docs/RESULT_PROTECT90_MEMBRANE_v0.1.md SS2.1)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import real_protect90_membrane_bridge as bridge


def _make_episode_df(n=6400, fs=6400, seed=0, jump_at=None):
    """Trzy zbalansowane fazy 50 Hz (baseline) -- opcjonalnie z 'skokiem'
    (dodatkowym wspoldzielonym skladnikiem burzacym symetrie) od probki
    `jump_at` -- imitujace zdarzenie zwarciowe w bardzo uproszczonej
    formie, wystarczajaco do przetestowania mechaniki pipeline'u."""
    rng = np.random.default_rng(seed)
    t = (np.arange(n) + 1) / fs
    data = {"time_s": t}
    angles = {1: 0.0, 2: -2 * np.pi / 3, 3: 2 * np.pi / 3}
    shared_shock = np.zeros(n)
    if jump_at is not None:
        shared_shock[jump_at:] = 5.0 * np.sin(2 * np.pi * 200 * t[jump_at:])
    for phase, angle in angles.items():
        base = 10 * np.sin(2 * np.pi * 50 * t + angle)
        noise = rng.normal(0, 0.05, n)
        cur = base + noise + shared_shock
        vol = 100 * np.sin(2 * np.pi * 50 * t + angle) + rng.normal(0, 0.05, n)
        data[f"{bridge.LOCATION}_cur_L{phase}_A"] = cur
        data[f"{bridge.LOCATION}_vol_L{phase}_V"] = vol
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_dataset(tmp_path, monkeypatch):
    data_root = tmp_path / "protect90"
    frozen_dir = data_root / "frozen"
    pkl_dir = frozen_dir / "preprocessed_data"
    raw_dir = data_root / "raw"
    pkl_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    sample_ids = [10, 11, 12, 13, 14]
    t_evnt_starts = {}
    for i, sid in enumerate(sample_ids):
        t_evnt = 0.3 + 0.01 * i
        jump_idx = int(round(t_evnt * 6400))
        df = _make_episode_df(seed=sid, jump_at=jump_idx)
        df.to_pickle(pkl_dir / f"{sid}_sample_hv_double_line_90kv.pkl")
        t_evnt_starts[sid] = t_evnt

    (frozen_dir / "B4_GRID_PROTECT90_FROZEN_EPISODE_IDS.json").write_text(
        json.dumps({
            "selection_salt": "test",
            "selection_rule": "smoke-test fixture",
            "per_fault_class": len(sample_ids),
            "sample_ids": sample_ids,
        }),
        encoding="utf-8",
    )

    labels = pd.DataFrame({
        "sample_id": sample_ids,
        "t_evnt_start": [t_evnt_starts[s] for s in sample_ids],
        "t_evnt_end": [t_evnt_starts[s] + 0.2 for s in sample_ids],
        "sc_type": [0, 1, 2, 3, 0],
    })
    labels.to_csv(raw_dir / "hv_double_line_90kv_labels.csv", index=False)

    raw_results_path = tmp_path / "protect90_membrane_raw_results.json"

    monkeypatch.setattr(bridge, "DATA_ROOT", data_root)
    monkeypatch.setattr(bridge, "FROZEN_DIR", frozen_dir)
    monkeypatch.setattr(bridge, "EPISODE_IDS_FILE", frozen_dir / "B4_GRID_PROTECT90_FROZEN_EPISODE_IDS.json")
    monkeypatch.setattr(bridge, "LABELS_FILE", raw_dir / "hv_double_line_90kv_labels.csv")
    monkeypatch.setattr(bridge, "PKL_DIR", pkl_dir)
    monkeypatch.setattr(bridge, "RAW_RESULTS_PATH", raw_results_path)

    return sample_ids


def test_pipeline_runs_end_to_end_on_small_sample(synthetic_dataset):
    progress = bridge.run_extraction(time_budget_s=60)
    assert progress["complete"] is True
    assert progress["n_done"] == len(synthetic_dataset)

    report = bridge.analyze()
    assert report["n_episodes_total"] == len(synthetic_dataset)
    for w in bridge.WINDOW_SIZES:
        for n_label in ("N3", "N6"):
            cell = report["windows"][str(w)][n_label]
            assert "spectral_concentration" in cell
            assert "participation_ratio" in cell
            assert "membrane_spectral_ratio" in cell

    classification = bridge.classify(report)
    assert set(classification.keys()) == {"N3", "N6"}
    for n_label, verdict in classification.items():
        assert verdict["status"] in {
            "SUPPORTED", "NOT_SUPPORTED", "NIESTABILNY", "INCONCLUSIVE",
            "NOT_SUPPORTED (kierunek przeciwny do przewidywania)",
        }


def test_run_extraction_is_resumable(synthetic_dataset):
    first = bridge.run_extraction(max_episodes=2, time_budget_s=60)
    assert first["n_done"] == 2
    assert first["complete"] is False

    second = bridge.run_extraction(time_budget_s=60)
    assert second["n_done"] == len(synthetic_dataset)
    assert second["complete"] is True


def test_event_start_index_matches_time_axis():
    time_s = (np.arange(6400) + 1) / 6400.0
    idx = bridge.event_start_index(time_s, 0.3)
    assert time_s[idx] >= 0.3
    if idx > 0:
        assert time_s[idx - 1] < 0.3


def test_paired_wilcoxon_test_sign_is_directionally_correct():
    """Regresja dla bledu znaku scipy.stats.wilcoxon opisanego w
    docs/RESULT_PROTECT90_MEMBRANE_v0.1.md SS2.1: gdy `event` jest
    SYSTEMATYCZNIE wyzsze niz `baseline`, `effect_size_r` MUSI byc
    dodatnie (nie ujemne, jak dawal surowy odczyt `scipy` zstatistic
    przed poprawka)."""
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, 60)
    event_higher = baseline + 2.0
    result_higher = bridge.paired_wilcoxon_test(event_higher, baseline)
    assert result_higher["effect_size_r"] > 0.3
    assert result_higher["pvalue"] < 0.05

    event_lower = baseline - 2.0
    result_lower = bridge.paired_wilcoxon_test(event_lower, baseline)
    assert result_lower["effect_size_r"] < -0.3
    assert result_lower["pvalue"] < 0.05


def test_paired_wilcoxon_test_inconclusive_on_tiny_sample():
    rng = np.random.default_rng(1)
    baseline = rng.normal(0, 1, 3)
    event = baseline + 1.0
    result = bridge.paired_wilcoxon_test(event, baseline)
    assert result["inconclusive"] is True
