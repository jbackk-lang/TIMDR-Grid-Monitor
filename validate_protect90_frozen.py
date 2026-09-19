"""Technical validation of the frozen PROTECT-90 selection.

This is deliberately not a B4 outcome analysis. It checks that every
pre-registered episode can be read and reduced without missing values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from protect90_adapter import load_protect90_episode


def main(data_dir: Path, selection_path: Path, output: Path) -> None:
    sample_ids = json.loads(selection_path.read_text(encoding="utf-8"))["sample_ids"]
    rows = []
    for index, sample_id in enumerate(sample_ids, start=1):
        signals, rate, provenance = load_protect90_episode(data_dir, sample_id)
        arrays = (signals.voltage, signals.frequency, signals.harmonics, signals.load)
        if rate != 100.0 or any(not np.isfinite(array).all() for array in arrays):
            raise RuntimeError(f"Niepoprawne dane po redukcji dla epizodu {sample_id}")
        rows.append({
            "sample_id": sample_id,
            "location": provenance["location"],
            "n_derived_samples": len(signals.voltage),
        })
        if index % 64 == 0 or index == len(sample_ids):
            print(f"Zweryfikowano {index}/{len(sample_ids)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Walidacja techniczna OK: {len(rows)} epizodów. Zapis: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/protect90/frozen")
    parser.add_argument("--selection", default="data/protect90/frozen/B4_GRID_PROTECT90_FROZEN_EPISODE_IDS.json")
    parser.add_argument("--output", default="data/protect90/frozen/TECHNICAL_VALIDATION.csv")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.selection), Path(args.output))
