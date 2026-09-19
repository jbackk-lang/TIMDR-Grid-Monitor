import pandas as pd

from freeze_protect90 import select_episode_ids


def test_selection_is_balanced_and_deterministic():
    labels = pd.DataFrame({
        "sample_id": list(range(1024)),
        "sc_type": [value // 256 for value in range(1024)],
    })
    first = select_episode_ids(labels)
    assert first == select_episode_ids(labels)
    assert len(first) == 512
    assert len(set(first)) == 512
