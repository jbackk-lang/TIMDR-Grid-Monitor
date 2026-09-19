"""Create a deterministic, pre-waveform B4-Grid-PROTECT90 frozen manifest.

Selection is stratified only by published fault category and uses a fixed
SHA-256 ordering. It never reads waveform pickle files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ARCHIVE_NAME = "hv_double_line_90kv_preprocessed_data.zip"
ARCHIVE_MD5 = "7cf176f169299b825ba6a6be102edca8"
LABELS_NAME = "hv_double_line_90kv_labels.csv"
SELECTION_SALT = "B4-GRID-PROTECT90-v1-prewaveform"
PER_FAULT_CLASS = 128


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_episode_ids(labels: pd.DataFrame) -> list[int]:
    required = {"sample_id", "sc_type"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Brak kolumn {sorted(required - set(labels.columns))}")
    selected: list[int] = []
    for fault_class in sorted(labels["sc_type"].unique()):
        candidates = labels.loc[labels["sc_type"] == fault_class, "sample_id"].astype(int).tolist()
        ordered = sorted(candidates, key=lambda value: hashlib.sha256(
            f"{SELECTION_SALT}:{fault_class}:{value}".encode("utf-8")
        ).hexdigest())
        if len(ordered) < PER_FAULT_CLASS:
            raise ValueError(f"Za mało epizodów klasy {fault_class}: {len(ordered)}")
        selected.extend(ordered[:PER_FAULT_CLASS])
    return sorted(selected)


def main(raw_dir: Path, output_dir: Path) -> None:
    labels_path = raw_dir / LABELS_NAME
    archive_path = raw_dir / ARCHIVE_NAME
    if not labels_path.is_file() or not archive_path.is_file():
        raise FileNotFoundError("Wymagane są pobrane labels CSV oraz pełne archiwum przebiegów.")
    if file_hash(archive_path, "md5") != ARCHIVE_MD5:
        raise RuntimeError("Archiwum nie przechodzi opublikowanej kontroli MD5.")
    labels = pd.read_csv(labels_path)
    selected = select_episode_ids(labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "B4_GRID_PROTECT90_FROZEN_EPISODE_IDS.json"
    selection_payload = {
        "selection_salt": SELECTION_SALT,
        "selection_rule": "128 episode IDs per sc_type, SHA-256 ordered before waveform access",
        "per_fault_class": PER_FAULT_CLASS,
        "sample_ids": selected,
    }
    selection_path.write_text(json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "state": "FROZEN",
        "dataset_id": "PROTECT-90 v1.0.0",
        "doi": "10.5281/zenodo.21109169",
        "source_type": "physically grounded EMT simulation; not field data",
        "waveform_archive": {
            "name": ARCHIVE_NAME,
            "md5": ARCHIVE_MD5,
            "sha256": file_hash(archive_path, "sha256"),
        },
        "labels_csv": {
            "name": LABELS_NAME,
            "sha256": file_hash(labels_path, "sha256"),
        },
        "sampling_rate_hz": 6400,
        "episode_duration_seconds": 1.0,
        "analysis_observability": "one protection-relevant location, selected before waveform access",
        "derived_grid_channels": {
            "voltage": "mean three-phase 10 ms RMS",
            "frequency": "phase-A zero-crossing estimate",
            "harmonics": "one-cycle FFT ratio in percent",
            "load": "apparent-power proxy sqrt(3)*Vrms*Irms",
        },
        "episode_selection_file": selection_path.name,
        "episode_selection_sha256": file_hash(selection_path, "sha256"),
        "n_selected_episodes": len(selected),
        "analysis_rule": "compare pre-event baseline with event-aligned windows; do not tune selection after results",
    }
    (output_dir / "B4_GRID_PROTECT90_FROZEN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Zamrożono {len(selected)} epizodów w {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/protect90/raw")
    parser.add_argument("--output-dir", default="data/protect90/frozen")
    args = parser.parse_args()
    main(Path(args.raw_dir), Path(args.output_dir))
