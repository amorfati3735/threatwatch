#!/usr/bin/env python3
"""
Standalone evaluation of a saved checkpoint on a preprocessed split.

Note on standardisation: a ``.npz`` produced by ``scripts/preprocess.py`` already
contains **standardised** windows (the scaler was fitted on train and applied to
everything).  This script therefore runs the network directly on the stored
tensors instead of re-applying the scaler, which would double-standardise them.
The live path is different on purpose: there the packets are raw, so
:class:`~threatwatch.ml.inference.SensorMLPredictor` applies the scaler itself.

Usage:
    python scripts/evaluate.py --checkpoint models/cnn_transformer.pt
    python scripts/evaluate.py --checkpoint models/cnn_transformer.pt --split val
    python scripts/evaluate.py --checkpoint models/cnn_transformer.pt --task binary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from threatwatch.config import (
    BINARY_CLASS_NAMES,
    CLASS_NAMES,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
)
from threatwatch.data.dataset import load_processed
from threatwatch.ml.metrics import compute_metrics, save_confusion_matrix, summarize


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=None)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--task", choices=["3class", "binary"], default="3class")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ds = args.dataset
    if ds is None:
        cands = sorted(PROCESSED_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        if not cands:
            raise SystemExit("No processed dataset found. Run scripts/preprocess.py first.")
        ds = cands[-1]
    d = load_processed(ds)
    print(f"[evaluate] checkpoint : {args.checkpoint}")
    print(f"[evaluate] dataset    : {ds}  split={args.split}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = ckpt["model_name"]
    n_channels = int(ckpt["n_channels"])
    class_names = list(ckpt.get("class_names", CLASS_NAMES))
    temperature = float(ckpt.get("temperature", 1.0) or 1.0)

    from threatwatch.models.seq_models import build_model

    model = build_model(model_name, n_channels, int(ckpt.get("n_classes", len(class_names))),
                        **(ckpt.get("model_kwargs") or {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X = d[f"{args.split}_X"]
    y = d[f"{args.split}_y"]

    if args.task == "binary":
        mask = y <= 1
        X, y = X[mask], y[mask]
        class_names = BINARY_CLASS_NAMES

    if X.shape[-1] != n_channels:
        print(f"[evaluate] WARNING: dataset has {X.shape[-1]} channels but the checkpoint "
              f"expects {n_channels}. Results will be meaningless.")

    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32)) / temperature
        probs = torch.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=1)

    metrics = compute_metrics(y, preds, class_names)
    print("\n" + summarize(metrics, f"{args.checkpoint.stem} [{args.split}]"))

    from threatwatch.ml.training import expected_calibration_error

    print(f"  temperature: {temperature:.3f}")
    print(f"  ECE        : {expected_calibration_error(probs, np.asarray(y)):.4f}")

    tag = args.tag or f"{args.checkpoint.stem}_{args.split}"
    save_confusion_matrix(metrics["confusion_matrix"], RESULTS_DIR / f"{tag}_confusion.png",
                          class_names, title=f"{args.checkpoint.stem} — {args.split}", normalize=True)
    (RESULTS_DIR / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[evaluate] wrote results/{tag}_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
