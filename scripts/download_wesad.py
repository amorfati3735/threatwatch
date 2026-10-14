#!/usr/bin/env python3
"""
Download and extract the real WESAD dataset.

The predecessor project hard-coded a **dead** ScieBo share token
(``mcrNsj7645odNqP``), so its downloader could never have produced the data its
README claimed.  This uses the working token and verifies the byte count.

Usage:
    python scripts/download_wesad.py                 # ~2.25 GB download + extract
    python scripts/download_wesad.py --dest data/wesad
    python scripts/download_wesad.py --no-extract
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from threatwatch.config import WESAD_DIR

# Verified working link (HTTP 206, content-type application/zip).
WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"
EXPECTED_BYTES = 2_249_444_501  # ~2.25 GB


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(dest: Path, resume: bool = True) -> Path:
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "WESAD.zip"

    if zip_path.exists() and zip_path.stat().st_size == EXPECTED_BYTES:
        print(f"[download] already complete: {zip_path} ({human(EXPECTED_BYTES)})")
        return zip_path

    start = zip_path.stat().st_size if (resume and zip_path.exists()) else 0
    headers = {"Range": f"bytes={start}-"} if start else {}

    print(f"[download] {WESAD_URL}")
    print(f"[download] -> {zip_path}" + (f" (resuming at {human(start)})" if start else ""))
    with requests.get(WESAD_URL, stream=True, timeout=60, headers=headers) as r:
        if r.status_code not in (200, 206):
            raise SystemExit(f"[download] unexpected HTTP {r.status_code}; the share link may have moved.")
        mode = "ab" if start else "wb"
        total = start + int(r.headers.get("content-length", 0))
        done = start
        with open(zip_path, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                pct = (done / total * 100) if total else 0
                print(f"\r[download] {human(done)} / {human(total)}  ({pct:5.1f}%)", end="")
    print()

    size = zip_path.stat().st_size
    if size != EXPECTED_BYTES:
        print(f"[download] WARNING: size {size} != expected {EXPECTED_BYTES}. "
              f"Re-run to resume, or verify manually.")
    else:
        print(f"[download] verified: {human(size)}")
    return zip_path


def extract(zip_path: Path, dest: Path) -> None:
    print(f"[extract] -> {dest}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    subjects = sorted(p.name for p in dest.glob("WESAD/S*") if p.is_dir())
    if not subjects:
        subjects = sorted(p.name for p in dest.glob("S*") if p.is_dir())
    print(f"[extract] done. subjects: {subjects}")


def main() -> int:
    p = argparse.ArgumentParser(description="Download and extract WESAD")
    p.add_argument("--dest", type=Path, default=WESAD_DIR)
    p.add_argument("--no-extract", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()

    zip_path = download(a.dest, resume=not a.no_resume)
    if not a.no_extract:
        extract(zip_path, a.dest)
        print("\n[download] next: python scripts/preprocess.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
