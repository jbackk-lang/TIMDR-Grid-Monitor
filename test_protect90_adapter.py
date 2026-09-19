import numpy as np
import pandas as pd

from protect90_adapter import load_protect90_episode


def test_protect90_adapter_derives_aligned_grid_channels(tmp_path):
    n = 6400
    t = np.arange(n) / 6400.0
    data = {"time_s": t}
    for phase, angle in enumerate((0.0, -2 * np.pi / 3, 2 * np.pi / 3), start=1):
        data[f"Bus_1_Line_01_02A_vol_L{phase}_V"] = 100 * np.sin(2 * np.pi * 50 * t + angle)
        data[f"Bus_1_Line_01_02A_cur_L{phase}_A"] = 10 * np.sin(2 * np.pi * 50 * t + angle)
    folder = tmp_path / "preprocessed_data"
    folder.mkdir()
    pd.DataFrame(data).to_pickle(folder / "7_sample_hv_double_line_90kv.pkl")
    signals, rate, provenance = load_protect90_episode(tmp_path, 7)
    assert rate == 100.0
    assert len(signals.voltage) == len(signals.frequency) == 100
    assert np.allclose(signals.frequency, 50.0)
    assert provenance["sample_id"] == 7
