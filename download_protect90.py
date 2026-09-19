"""Download and verify the public PROTECT-90 release from Zenodo."""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

BASE_URL = "https://zenodo.org/records/21109169/files"
FILES = {
    "hv_double_line_90kv_labels.csv": "5f015330f77ed53b76bd5db26e83c48d",
    "README.md": "bc10613ec00d5d659ac59d2b0d0c5800",
    "hv_double_line_90kv_preprocessed_data.zip": "7cf176f169299b825ba6a6be102edca8",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / name
    urllib.request.urlretrieve(f"{BASE_URL}/{name}?download=1", target)
    if md5(target) != FILES[name]:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Błędna suma MD5 dla {name}")
    print(f"OK: {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/protect90/raw")
    parser.add_argument("--full", action="store_true", help="pobierz również archiwum przebiegów około 12 GB")
    args = parser.parse_args()
    out = Path(args.output)
    download("README.md", out)
    download("hv_double_line_90kv_labels.csv", out)
    if args.full:
        download("hv_double_line_90kv_preprocessed_data.zip", out)
    else:
        print("Pobrano metadane. Użyj --full, aby pobrać przebiegi.")
