#!/usr/bin/env python3
"""
Extract model-ready windows from real WESAD and save leakage-free datasets.

Efficiency note: the raw WESAD pickles are ~13 GB, so each subject is loaded
**once** and windowed for every requested channel set.  Producing the ``full``
and ``hw`` variants therefore costs one pass over the data, not two.

Leakage guarantees (see docs/FLAWS_AND_FIXES.md):
  * subjects are partitioned into train/val/test before any windowing;
  * the standardizer is fitted on TRAIN windows only;
  * every window's majority class must dominate 70% of the window.

Usage:
    python scripts/preprocess.py                          # both channel sets, 15s/5s
    python scripts/preprocess.py --channels full
    python scripts/preprocess.py --window 30 --stride 10
    python scripts/preprocess.py --data-dir /path/to/wesad
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from threatwatch.config import (
    CHANNEL_SETS,
    MODELS_DIR,
    PROCESSED_DIR,
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    WESAD_DIR,
    WindowConfig,
)
from threatwatch.data.dataset import Standardizer, extract_windows, save_processed
from threatwatch.data.wesad import available_subjects, extract_subject

SPLITS = [("train", TRAIN_SUBJECTS), ("val", VAL_SUBJECTS), ("test", TEST_SUBJECTS)]


def parse_args():
    p = argparse.ArgumentParser(description="WESAD preprocessing / windowing")
    p.add_argument("--data-dir", type=Path, default=WESAD_DIR)
    p.add_argument("--window", type=float, default=15.0, help="window length (seconds)")
    p.add_argument("--stride", type=float, default=5.0, help="stride (seconds)")
    p.add_argument("--fs", type=float, default=32.0, help="target sampling rate (Hz)")
    p.add_argument("--channels", choices=["full", "hw", "both"], default="both")
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    wcfg = WindowConfig(window_sec=args.window, stride_sec=args.stride, target_fs=args.fs)

    keys: List[str] = ["full", "hw"] if args.channels == "both" else [args.channels]
    channel_sets = {k: CHANNEL_SETS[k] for k in keys}

    found = available_subjects(args.data_dir)
    print(f"[preprocess] data dir : {args.data_dir}")
    print(f"[preprocess] subjects : {len(found)} -> {found}")
    if not found:
        print("[preprocess] ERROR: no WESAD pickles found. Run scripts/download_wesad.py "
              "or place S*/S*.pkl under data/wesad/.")
        return 1

    missing = [s for s in (TRAIN_SUBJECTS + VAL_SUBJECTS + TEST_SUBJECTS) if s not in found]
    if missing:
        print(f"[preprocess] WARNING: missing expected subjects: {missing}")

    print(f"[preprocess] window   : {wcfg.window_sec:g}s / stride {wcfg.stride_sec:g}s "
          f"@ {wcfg.target_fs:g} Hz -> {wcfg.window_samples} samples")
    print(f"[preprocess] channels : " + ", ".join(f"{k}={v}" for k, v in channel_sets.items()))

    # bucket[channel_key][split] -> lists
    acc: Dict[str, Dict[str, Dict[str, list]]] = {
        k: {s: {"X": [], "y": [], "subjects": [], "starts": []} for s, _ in SPLITS} for k in keys
    }

    t0 = time.time()
    for split_name, subjects in SPLITS:
        print(f"\n[preprocess] --- {split_name.upper()} subjects: {subjects} ---")
        for subj in subjects:
            data = extract_subject(subj, args.data_dir, target_fs=wcfg.target_fs)
            if data is None:
                print(f"  [skip] {subj}: not found")
                continue
            minutes = data["n"] / wcfg.target_fs / 60.0
            for key, channels in channel_sets.items():
                X, y, starts = extract_windows(
                    data, wcfg.window_samples, wcfg.stride_samples, feature_channels=channels
                )
                if len(y) == 0:
                    continue
                bucket = acc[key][split_name]
                bucket["X"].append(X)
                bucket["y"].append(y)
                bucket["subjects"].append(np.array([subj] * len(y)))
                bucket["starts"].append(starts)
                dist = np.bincount(y, minlength=3)
                print(f"  {subj} [{key:4s}] {minutes:5.1f} min -> {len(y):5d} windows  "
                      f"base={dist[0]} stress={dist[1]} amuse={dist[2]}")
            del data

    print(f"\n[preprocess] extraction took {time.time() - t0:.1f}s")

    scalers = {}
    for key in keys:
        splits = {}
        for split_name, _ in SPLITS:
            b = acc[key][split_name]
            if not b["y"]:
                print(f"[preprocess] ERROR: no windows for channel set '{key}' / split '{split_name}'")
                return 1
            splits[split_name] = {
                "X": np.concatenate(b["X"], axis=0),
                "y": np.concatenate(b["y"], axis=0),
                "subjects": np.concatenate(b["subjects"], axis=0),
                "starts": np.concatenate(b["starts"], axis=0),
            }

        # Fit the standardizer on TRAIN ONLY, then apply it everywhere.
        scaler = Standardizer(channels=tuple(channel_sets[key])).fit(splits["train"]["X"])
        for split_name, _ in SPLITS:
            splits[split_name]["X"] = scaler.transform(splits[split_name]["X"])
        scalers[key] = scaler

        out = args.out_dir / f"{wcfg.tag(key)}.npz"
        save_processed(out, splits["train"], splits["val"], splits["test"], wcfg, channels=channel_sets[key])

        scaler_path = MODELS_DIR / ("scaler.json" if key == "full" else f"scaler_{key}.json")
        scaler_path.write_text(json.dumps(scaler.to_dict(), indent=2))

        print(f"\n[preprocess] ==== {key} ====")
        for split_name, _ in SPLITS:
            s = splits[split_name]
            dist = np.bincount(s["y"], minlength=3)
            print(f"  {split_name:5s}: {len(s['y']):6d} windows  X={s['X'].shape}  "
                  f"base={dist[0]} stress={dist[1]} amuse={dist[2]}")
        print(f"  dataset: {out}")
        print(f"  scaler : {scaler_path}")

    print(f"\n[preprocess] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
