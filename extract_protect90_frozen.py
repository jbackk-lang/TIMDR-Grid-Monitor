"""Extract only the pre-registered PROTECT-90 episodes from the ZIP archive."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def main(archive: Path, selection: Path, output: Path) -> None:
    ids = json.loads(selection.read_text(encoding="utf-8"))["sample_ids"]
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        for index, sample_id in enumerate(ids, start=1):
            name = f"hv_double_line_90kv_preprocessed_data/{sample_id}_sample_hv_double_line_90kv.pkl"
            if name not in names:
                raise FileNotFoundError(f"Archiwum nie zawiera {name}")
            target = output / Path(name).name
            if not target.exists():
                with zipped.open(name) as source, target.open("wb") as destination:
                    destination.write(source.read())
            if index % 32 == 0 or index == len(ids):
                print(f"Wyodrębniono {index}/{len(ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="data/protect90/raw/hv_double_line_90kv_preprocessed_data.zip")
    parser.add_argument("--selection", default="data/protect90/frozen/B4_GRID_PROTECT90_FROZEN_EPISODE_IDS.json")
    parser.add_argument("--output", default="data/protect90/frozen/preprocessed_data")
    args = parser.parse_args()
    main(Path(args.archive), Path(args.selection), Path(args.output))
