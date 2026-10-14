#!/usr/bin/env python3
"""
Computational-feasibility benchmark for every architecture, measured not guessed.

The brief asks for the *feasibility* of each model on the available hardware to be
established before committing to a training configuration.  This script reports,
for each architecture and window length: parameter count, forward+backward time
per batch, peak memory, and the projected wall-clock for one epoch on the real
training-set size.

    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --seconds 15 10 --batch 64 16
    python scripts/benchmark_models.py --n-train-windows 2156
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from threatwatch.config import (
    AUTO_BATCH,
    NUM_CLASSES,
    PACKET_CHANNELS,
    RESULTS_DIR,
    TARGET_FS,
)
from threatwatch.models.seq_models import build_model, count_parameters
from threatwatch.utils.logging import get_logger

log = get_logger("benchmark")

MODELS = ["cnn", "bilstm", "transformer", "cnn_transformer", "patch_vit"]


def bench_one(name: str, T: int, C: int, B: int, device, iters: int = 5):
    """Return (n_params, seconds_per_batch, peak_mb) for one (model, shape)."""
    model = build_model(name, C, NUM_CLASSES).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(B, T, C, device=device)
    y = torch.randint(0, NUM_CLASSES, (B,), device=device)

    for _ in range(2):  # warm-up (cuDNN autotune / cache effects)
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_batch = (time.perf_counter() - t0) / iters

    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
    return count_parameters(model), per_batch, peak_mb


def parse_args():
    p = argparse.ArgumentParser(description="ThreatWatch architecture benchmark")
    p.add_argument("--seconds", type=float, nargs="+", default=[15.0, 10.0],
                   help="candidate window lengths in seconds")
    p.add_argument("--batch", type=int, nargs="+", default=None,
                   help="batch sizes (default: config AUTO_BATCH values)")
    p.add_argument("--channels", type=int, default=len(PACKET_CHANNELS))
    p.add_argument("--n-train-windows", type=int, default=2156,
                   help="real training-window count, for the epoch estimate")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "benchmark.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    batches = args.batch or sorted(set(AUTO_BATCH.values()))

    print(f"device  : {device}   (torch {torch.__version__}, {platform.processor() or platform.machine()})")
    print(f"windows : {args.n_train_windows} training windows × {args.channels} channels @ {TARGET_FS:g} Hz\n")

    header = f"{'model':<16}{'sec':>5}{'T':>6}{'B':>5}{'params':>11}{'ms/batch':>11}{'epoch':>10}{'peakMB':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    for sec in args.seconds:
        T = int(round(sec * TARGET_FS))
        for B in batches:
            for name in MODELS:
                try:
                    p, per_batch, peak = bench_one(name, T, args.channels, B, device)
                except RuntimeError as exc:
                    print(f"{name:<16}{sec:>5.0f}{T:>6}{B:>5}   FAILED: {str(exc)[:52]}")
                    rows.append({"model": name, "window_sec": sec, "T": T, "batch": B,
                                 "error": str(exc)[:160]})
                    continue
                epoch_min = per_batch * max(1, args.n_train_windows // B) / 60.0
                print(f"{name:<16}{sec:>5.0f}{T:>6}{B:>5}{p:>11,}"
                      f"{per_batch * 1000:>11.1f}{epoch_min:>9.2f}m{peak:>9.0f}")
                rows.append({
                    "model": name, "window_sec": sec, "T": T, "batch": B,
                    "params": p, "ms_per_batch": round(per_batch * 1000, 2),
                    "epoch_min_est": round(epoch_min, 3), "peak_mb": round(peak, 1),
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "device": str(device),
        "torch": torch.__version__,
        "channels": args.channels,
        "target_fs": TARGET_FS,
        "n_train_windows": args.n_train_windows,
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
