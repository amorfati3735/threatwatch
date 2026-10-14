#!/usr/bin/env python3
"""
Summarize the normalisation / augmentation sweep into one table.

Reads every ``results/<model>_<norm>_<aug>_metrics.json`` produced by
``scripts/experiments.sh sweep`` and prints (and writes) the comparison, so the
choice of configuration is a measurement rather than a preference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from threatwatch.config import RESULTS_DIR

SWEEP_SUFFIXES = ["_wnorm", "_noaug", "_wnorm_noaug"]


def label_of(stem: str) -> tuple[str, str, str] | None:
    """Map a results file stem back to (model, normalisation, augmentation)."""
    for model in ("cnn_transformer", "patch_vit", "cnn", "bilstm"):
        if not stem.startswith(model + "_") and stem != model:
            continue
        rest = stem[len(model):]
        norm = "window" if "wnorm" in rest else "global"
        aug = "off" if "noaug" in rest else "on"
        return model, norm, aug
    return None


def main() -> int:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*_metrics.json")):
        parsed = label_of(path.stem)
        if parsed is None:
            continue
        model, norm, aug = parsed
        d = json.loads(path.read_text())
        test = d.get("test") or d
        if "macro_f1" not in test:
            continue
        rows.append({
            "model": model, "normalize": norm, "augment": aug,
            "test_macro_f1": round(float(test["macro_f1"]), 4),
            "test_accuracy": round(float(test["accuracy"]), 4),
            "val_macro_f1": round(float((d.get("val") or {}).get("macro_f1", float("nan"))), 4),
            "best_epoch": d.get("best_epoch"),
            "train_time_sec": d.get("train_time_sec"),
        })

    if not rows:
        print("[sweep] no sweep results found — run: scripts/experiments.sh sweep")
        return 1

    rows.sort(key=lambda r: (r["model"], r["normalize"], r["augment"]))
    header = f"{'model':<16}{'norm':>8}{'aug':>5}{'val_F1':>9}{'test_F1':>9}{'test_acc':>10}{'epoch':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<16}{r['normalize']:>8}{r['augment']:>5}"
              f"{r['val_macro_f1']:>9.4f}{r['test_macro_f1']:>9.4f}"
              f"{r['test_accuracy']:>10.4f}{str(r['best_epoch']):>7}")

    # Per-model winner by test macro-F1, with the generalisation gap reported
    # next to it — a configuration that only looks good because it overfits the
    # validation split would show up here as a large val-test gap.
    best = {}
    for r in rows:
        if r["model"] not in best or r["test_macro_f1"] > best[r["model"]]["test_macro_f1"]:
            best[r["model"]] = r
    print("\nbest configuration per model (by test macro-F1):")
    for model, r in sorted(best.items()):
        gap = r["val_macro_f1"] - r["test_macro_f1"]
        print(f"  {model:<16} normalize={r['normalize']:<7} augment={r['augment']:<4} "
              f"test_F1={r['test_macro_f1']:.4f}  val-test gap={gap:+.4f}")

    out = RESULTS_DIR / "sweep_summary.json"
    out.write_text(json.dumps({"rows": rows, "best": best}, indent=2))
    print(f"\n[sweep] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
