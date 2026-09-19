"""Stage a small, redistributable PROTECT-90 example set for the dashboard.

The source archive remains outside Git. This script copies exactly two already
frozen episode IDs for each fault class, records their IDs and SHA-256 hashes,
and never changes the full frozen benchmark selection.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
FROZEN = ROOT / "data" / "protect90" / "frozen"
DESTINATION = ROOT / "data" / "protect90" / "examples" / "preprocessed_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    selected = set(json.loads(
        (FROZEN / "B4_GRID_PROTECT90_FROZEN_EPISODE_IDS.json").read_text(encoding="utf-8")
    )["sample_ids"])
    labels = pd.read_csv(ROOT / "data" / "protect90" / "raw" / "hv_double_line_90kv_labels.csv")
    chosen = []
    for fault_class in sorted(labels["sc_type"].unique()):
        ids = sorted(
            int(value) for value in labels.loc[labels["sc_type"] == fault_class, "sample_id"]
            if int(value) in selected
        )
        chosen.extend(ids[:2])
    if len(chosen) != 8:
        raise RuntimeError(f"Oczekiwano 8 przykładów, otrzymano {chosen}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    records = []
    for sample_id in chosen:
        source = FROZEN / "preprocessed_data" / f"{sample_id}_sample_hv_double_line_90kv.pkl"
        target = DESTINATION / source.name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target)
        records.append({"sample_id": sample_id, "filename": target.name, "sha256": sha256(target)})
    manifest = {
        "dataset": "PROTECT-90 v1.0.0",
        "doi": "10.5281/zenodo.21109169",
        "license": "CC-BY-4.0",
        "selection_rule": "two lowest frozen sample IDs per sc_type; display examples only",
        "n_examples": len(records),
        "examples": records,
    }
    (DESTINATION.parent / "EXAMPLE_SET_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Przygotowano {len(records)} przykładów w {DESTINATION.parent}")


if __name__ == "__main__":
    main()
