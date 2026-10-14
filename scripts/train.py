#!/usr/bin/env python3
"""
Train baselines and deep sequence models on a preprocessed WESAD dataset.

Both evaluation protocols from the brief are supported:

  3-class   baseline / stress / amusement   (the WESAD-native task)
  binary    stress vs baseline              (the literature-standard task)

Examples:
    python scripts/train.py --task 3class --models random_forest logistic
    python scripts/train.py --task 3class --models cnn bilstm transformer cnn_transformer patch_vit
    python scripts/train.py --task binary  --models random_forest cnn_transformer
    python scripts/train.py --channels hw  --models logistic        # hardware-only ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from threatwatch.config import (
    AUTO_BATCH,
    BINARY_CLASS_NAMES,
    CLASS_NAMES,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    TrainConfig,
)
from threatwatch.data.dataset import load_processed
from threatwatch.ml.metrics import compute_metrics, save_confusion_matrix, summarize
from threatwatch.ml.training import train_deep
from threatwatch.models.baselines import build_baseline, extract_stat_features

DEEP_MODELS = ["cnn", "bilstm", "transformer", "cnn_transformer", "patch_vit"]
# ``svm`` and ``naive_bayes`` are the base paper's own classifiers and are kept in
# the list so the comparison against it is a real measurement, not a comparison
# against a textbook example.
BASELINES = ["svm", "naive_bayes", "random_forest", "logistic"]
EVAL_PROTOCOLS = ["subject", "window_kfold"]


def parse_args():
    p = argparse.ArgumentParser(description="Train ThreatWatch classifiers")
    p.add_argument("--dataset", type=Path, default=None, help="processed .npz (default: newest)")
    p.add_argument("--models", nargs="+", default=DEEP_MODELS + BASELINES,
                   help=f"any of {DEEP_MODELS + BASELINES}")
    p.add_argument("--task", choices=["3class", "binary"], default="3class")
    p.add_argument("--channels", choices=["auto", "full", "hw"], default="auto",
                   help="which scaler to embed in the checkpoint")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=0, help="0 = per-model default")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--no-augment", action="store_true",
                   help="disable on-the-fly window augmentation")
    p.add_argument("--normalize", choices=["global", "window"], default="global",
                   help="'global' = training-set statistics; 'window' = per-window z-score "
                        "(subject-agnostic and reproducible on the ESP32)")
    p.add_argument("--protocol", choices=EVAL_PROTOCOLS, default="subject",
                   help="'subject' = honest subject-disjoint test set; "
                        "'window_kfold' = window-level stratified k-fold, which is the "
                        "methodology of the base paper (Wagh et al. 2025)")
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--model-kwargs", type=str, default=None,
                   help="JSON of architecture kwargs, e.g. '{\"width\": 32}' for a "
                        "deployment-sized CNN")
    p.add_argument("--tag", type=str, default=None,
                   help="extra tag appended to result/checkpoint names")
    return p.parse_args()


def window_normalize(X: np.ndarray) -> np.ndarray:
    """Z-score every window against itself, per channel.  Uses no dataset-level
    statistics, so applying it to val/test is not leakage."""
    X = np.nan_to_num(X, nan=0.0)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32)


def run_window_kfold(args, X, y, scaler, channels, window):
    """Window-level stratified k-fold — the base paper's evaluation protocol.

    This deliberately mixes subjects across folds, so windows from the same person
    appear in both train and test.  It is included *only* to reproduce the base
    paper's setup for comparison; it is not the project's headline result.
    """
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    class_names = BINARY_CLASS_NAMES if args.task == "binary" else CLASS_NAMES

    folds = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        print(f"\n[kfold:{args.protocol}] fold {fold}/{args.kfold}  "
              f"train={len(ytr)} test={len(yte)}")
        for name in args.models:
            if name in BASELINES:
                Ftr, Fte = extract_stat_features(Xtr), extract_stat_features(Xte)
                clf = build_baseline(name, seed=args.seed)
                clf.fit(Ftr, ytr)
                pred = clf.predict(Fte)
            elif name in DEEP_MODELS:
                from threatwatch.data.dataset import WindowDataset
                from threatwatch.ml.training import train_deep

                bs = args.batch_size or AUTO_BATCH.get(name, 32)
                tcfg = TrainConfig(
                    epochs=args.epochs, batch_size=bs, lr=args.lr, patience=args.patience,
                    device=args.device, seed=args.seed, tensorboard=False,
                    augment=not args.no_augment, normalize=args.normalize,
                )
                # train on the fold's train split, select on a slice of it, score on test
                cut = int(len(ytr) * 0.85)
                suffix_b = "" if args.task == "3class" else "_binary"
                res = train_deep(
                    name, (Xtr[:cut], ytr[:cut]), (Xtr[cut:], ytr[cut:]), (Xte, yte),
                    tcfg, tag=f"kfold{fold}_{name}{suffix_b}",
                    channels=channels, window=window,
                    scaler=scaler, class_names=class_names,
                )
                pred = np.asarray(res["test"]["y_pred"])
            else:
                continue

            folds.append({
                "fold": fold, "model": name,
                "macro_f1": float(f1_score(yte, pred, average="macro")),
                "accuracy": float(accuracy_score(yte, pred)),
            })

    summary = {}
    for name in args.models:
        rows = [f for f in folds if f["model"] == name]
        if not rows:
            continue
        f1s = np.array([r["macro_f1"] for r in rows], dtype=float)
        accs = np.array([r["accuracy"] for r in rows], dtype=float)
        summary[name] = {
            "macro_f1_mean": float(f1s.mean()), "macro_f1_std": float(f1s.std()),
            "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
            "folds": len(rows),
        }
        print(f"  {name:18s} macroF1={f1s.mean():.4f} ± {f1s.std():.4f}   "
              f"acc={accs.mean():.4f} ± {accs.std():.4f}")

    out = RESULTS_DIR / f"kfold{args.kfold}{'' if args.task == '3class' else '_binary'}.json"
    out.write_text(json.dumps({
        "protocol": "window_kfold (subject-mixing; base-paper methodology)",
        "n_windows": int(len(y)),
        "kfold": args.kfold,
        "task": args.task,
        "channels": channels,
        "normalize": args.normalize,
        "summary": summary,
        "folds": folds,
    }, indent=2))
    print(f"\n[train] wrote {out}")
    return 0


def resolve_dataset(path, channels: str) -> Path:
    if path is not None:
        return path
    cands = sorted(PROCESSED_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit("No processed dataset found. Run scripts/preprocess.py first.")
    if channels != "auto":
        want = f"_{channels}.npz"
        for c in cands:
            if c.name.endswith(want):
                return c
        print(f"[train] no dataset matching '{channels}' — using newest ({cands[-1].name})")
    return cands[-1]


def to_binary(X, y, subjects):
    """Keep baseline(0) and stress(1); drop amusement."""
    mask = y <= 1
    return X[mask], y[mask], subjects[mask]


def build_scaler_dict(models_dir: Path, channels: List[str]) -> Optional[dict]:
    """Load the persisted standardizer whose channel list matches these channels.

    Matching on the channel *names* rather than on counts means the full (7-channel)
    and hardware-only (6-channel) scalers can never be confused for one another.
    """
    for path in sorted(models_dir.glob("scaler*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        if list(d.get("channels", [])) == list(channels):
            return d
    return None


def dataset_channels(d) -> List[str]:
    ch = d.get("channels")
    if ch is None:
        return []
    return [str(c) for c in (ch.tolist() if hasattr(ch, "tolist") else ch)]


def main() -> int:
    args = parse_args()
    ds_path = resolve_dataset(args.dataset, args.channels)
    print(f"[train] dataset: {ds_path}")
    d = load_processed(ds_path)

    channels = dataset_channels(d)
    window = {
        "window_sec": float(d.get("w_window_sec", 15.0)),
        "stride_sec": float(d.get("w_stride_sec", 5.0)),
        "target_fs": float(d.get("w_target_fs", 32.0)),
    }
    print(f"[train] channels : {channels}")
    print(f"[train] window   : {window}")

    X_tr, y_tr = d["train_X"], d["train_y"]
    X_va, y_va = d["val_X"], d["val_y"]
    X_te, y_te = d["test_X"], d["test_y"]

    if args.task == "binary":
        X_tr, y_tr, _ = to_binary(X_tr, y_tr, d["train_subjects"])
        X_va, y_va, _ = to_binary(X_va, y_va, d["val_subjects"])
        X_te, y_te, _ = to_binary(X_te, y_te, d["test_subjects"])
        class_names = BINARY_CLASS_NAMES
        print("[train] task: binary (stress vs baseline)")
    else:
        class_names = CLASS_NAMES
        print("[train] task: 3-class")

    print(f"[train] shapes train={X_tr.shape} val={X_va.shape} test={X_te.shape}")

    # ---- normalisation mode ------------------------------------------------ #
    scaler = build_scaler_dict(MODELS_DIR, channels)
    if args.normalize == "window":
        # Per-window z-scoring needs no dataset statistics and cancels each
        # subject's personal offset.  Applied identically to every split, so it
        # is leakage-free (it only ever looks at the window it is given).
        X_tr, X_va, X_te = window_normalize(X_tr), window_normalize(X_va), window_normalize(X_te)
        print("[train] normalisation: per-window z-score (subject-agnostic)")
    elif scaler is not None:
        print(f"[train] normalisation: global, embedding scaler for {len(scaler.get('mean', []))} channels")
    else:
        print("[train] WARNING: no matching scaler.json found — checkpoint will not embed one")
    if scaler is None:
        scaler = {"mean": None, "std": None, "channels": list(channels), "mode": args.normalize}
    else:
        scaler = dict(scaler)
        scaler["mode"] = args.normalize

    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else None
    if model_kwargs:
        print(f"[train] model_kwargs: {model_kwargs}")

    if args.tag:
        # An explicit tag replaces the derived suffix, so a deliberately-tagged run
        # (e.g. the deployment-sized model) gets a clean, stable artifact name.
        suffix = f"_{args.tag}"
    else:
        suffix = "" if args.task == "3class" else "_binary"
        if args.normalize == "window":
            suffix += "_wnorm"
        if args.no_augment:
            suffix += "_noaug"

    # ---- the base paper's protocol (subject-mixing window k-fold) ---------- #
    if args.protocol == "window_kfold":
        X_all = np.concatenate([X_tr, X_va, X_te], axis=0)
        y_all = np.concatenate([y_tr, y_va, y_te], axis=0)
        print(f"[train] protocol: window-level {args.kfold}-fold (subject-mixing)")
        return run_window_kfold(args, X_all, y_all, scaler, channels, window)

    comparison = {}

    # ---- classical baselines ---------------------------------------------- #
    for name in [m for m in args.models if m in BASELINES]:
        t0 = time.time()
        Ftr, Fva, Fte = (extract_stat_features(X_tr), extract_stat_features(X_va), extract_stat_features(X_te))
        clf = build_baseline(name, seed=args.seed)
        clf.fit(Ftr, y_tr)
        metrics = compute_metrics(y_te, clf.predict(Fte), class_names)
        tag = f"baseline_{name}{suffix}"
        save_confusion_matrix(metrics["confusion_matrix"], RESULTS_DIR / f"{tag}_confusion.png",
                              class_names, title=f"{name} — test", normalize=True)
        (RESULTS_DIR / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
        print("\n" + summarize(metrics, f"BASELINE {name} (test)"))
        comparison[tag] = {
            "test_macro_f1": metrics["macro_f1"],
            "test_accuracy": metrics["accuracy"],
            "train_time_sec": round(time.time() - t0, 2),
        }

    # ---- deep models ------------------------------------------------------- #
    for name in [m for m in args.models if m in DEEP_MODELS]:
        bs = args.batch_size or AUTO_BATCH.get(name, 32)
        tcfg = TrainConfig(
            epochs=args.epochs, batch_size=bs, lr=args.lr, patience=args.patience,
            device=args.device, seed=args.seed, max_train_windows=args.max_train_windows,
            tensorboard=not args.no_tensorboard,
            augment=not args.no_augment, normalize=args.normalize,
        )
        tag = f"{name}{suffix}"
        print(f"\n[train] {name}: batch_size={bs}")
        res = train_deep(
            name, (X_tr, y_tr), (X_va, y_va), (X_te, y_te), tcfg, tag=tag,
            model_kwargs=model_kwargs,
            channels=channels, window=window, scaler=scaler, class_names=class_names,
        )
        print("\n" + summarize(res["test"], f"{tag} (test)"))
        comparison[tag] = {
            "params": res["params"],
            "test_macro_f1": res["test"]["macro_f1"],
            "test_accuracy": res["test"]["accuracy"],
            "train_time_sec": res["train_time_sec"],
            "inference_latency_ms": res["inference_latency_ms"],
            "temperature": res["temperature"],
        }

    out = RESULTS_DIR / f"comparison{suffix}.json"
    out.write_text(json.dumps(comparison, indent=2))
    print(f"\n[train] wrote comparison -> {out}")
    print("\n[train] ==== leaderboard (test macro-F1) ====")
    for name, row in sorted(comparison.items(), key=lambda kv: -kv[1]["test_macro_f1"]):
        print(f"  {name:30s} macroF1={row['test_macro_f1']:.4f}  acc={row['test_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
