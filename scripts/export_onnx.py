#!/usr/bin/env python3
"""
Export a trained checkpoint to ONNX (verified against PyTorch) for edge tooling.

    python scripts/export_onnx.py                          # newest checkpoint
    python scripts/export_onnx.py --checkpoint models/cnn_transformer.pt
    python scripts/export_onnx.py --all                    # export every checkpoint

The resulting .onnx opens in Netron and loads in onnxruntime.  See
``threatwatch/ml/export.py`` for the documented PyTorch -> ONNX -> TFLite route.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from threatwatch.config import MODELS_DIR
from threatwatch.ml.export import TFLITE_NOTE, export_onnx
from threatwatch.ml.inference import load_latest_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Export checkpoints to ONNX")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--all", action="store_true", help="export every .pt in models/")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--show-tflite-note", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.all:
        checkpoints = sorted(MODELS_DIR.glob("*.pt"))
    elif args.checkpoint is not None:
        checkpoints = [args.checkpoint]
    else:
        latest = load_latest_checkpoint()
        checkpoints = [latest] if latest else []

    if not checkpoints:
        print("No checkpoint found. Train one first: python scripts/train.py")
        return 1

    failures = 0
    for ckpt in checkpoints:
        try:
            path, report = export_onnx(ckpt, args.out, args.opset, verify=not args.no_verify)
            print(f"\n[export] {ckpt.name} -> {path}")
            for k, v in report.items():
                print(f"    {k}: {v}")
        except Exception as exc:
            failures += 1
            print(f"\n[export] {ckpt.name} FAILED: {exc}")

    if args.show_tflite_note:
        print("\n" + TFLITE_NOTE)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
