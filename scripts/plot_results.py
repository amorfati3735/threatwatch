#!/usr/bin/env python3
"""
Build the write-up figures from whatever is in ``results/``.

Produces:
  results/model_comparison.md   markdown table (3-class + binary)
  results/model_comparison.png  macro-F1 bar chart per protocol
  results/per_class_f1.png      per-class F1 grouped bars
  results/training_curves.png   val macro-F1 and loss per deep model

Usage:
    python scripts/plot_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from threatwatch.config import RESULTS_DIR


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def collect():
    """Return {protocol: {model: summary}} plus per-class detail."""
    protocols = {"3class": {}, "binary": {}}
    details = {}

    for path in sorted(RESULTS_DIR.glob("*_metrics.json")):
        data = load_json(path)
        if not data or "test" not in data:
            continue
        name = data.get("tag") or path.stem.replace("_metrics", "")
        protocol = "binary" if name.endswith("_binary") else "3class"
        clean = name.replace("_binary", "").replace("baseline_", "")
        protocols[protocol][clean] = {
            "macro_f1": data["test"]["macro_f1"],
            "accuracy": data["test"]["accuracy"],
            "params": data.get("params"),
            "latency_ms": data.get("inference_latency_ms"),
            "train_time": data.get("train_time_sec"),
            "temperature": data.get("temperature"),
        }
        details[clean] = data["test"].get("per_class", {})

    for protocol in protocols:
        comp = load_json(RESULTS_DIR / f"comparison{'' if protocol == '3class' else '_binary'}.json")
        if comp:
            for name, row in comp.items():
                clean = name.replace("_binary", "").replace("baseline_", "")
                protocols[protocol].setdefault(clean, {})
                protocols[protocol][clean].setdefault("macro_f1", row.get("test_macro_f1"))
                protocols[protocol][clean].setdefault("accuracy", row.get("test_accuracy"))
    return protocols, details


def markdown_table(protocols) -> str:
    lines = ["# Model comparison (real WESAD, subject-independent split)\n"]
    for protocol, models in protocols.items():
        if not models:
            continue
        header = "## 3-class (baseline / stress / amusement)" if protocol == "3class" \
            else "## Binary (stress vs baseline)"
        lines.append("\n" + header + "\n")
        lines.append("| model | accuracy | macro-F1 | params | latency (ms) | train (s) |")
        lines.append("|---|---|---|---|---|---|")
        for name, row in sorted(models.items(), key=lambda kv: -(kv[1].get("macro_f1") or 0)):
            acc = row.get("accuracy")
            f1 = row.get("macro_f1")
            params = row.get("params")
            lat = row.get("latency_ms")
            tt = row.get("train_time")
            lines.append(
                f"| {name} | {acc:.4f} | {f1:.4f} | "
                f"{'—' if not params else format(int(params), ',')} | "
                f"{'—' if lat is None else f'{lat:.2f}'} | "
                f"{'—' if tt is None else f'{tt:.1f}'} |"
            )
    return "\n".join(lines) + "\n"


def bar_chart(protocols, out: Path) -> None:
    present = {k: v for k, v in protocols.items() if v}
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(5.6 * len(present), 4.4), dpi=150, squeeze=False)
    for ax, (protocol, models) in zip(axes[0], present.items()):
        names = sorted(models, key=lambda n: -(models[n].get("macro_f1") or 0))
        vals = [models[n].get("macro_f1") or 0 for n in names]
        colors = ["#2b6cb0" if n in ("logistic", "random_forest") else "#2f855a" for n in names]
        ax.barh(range(len(names))[::-1], vals, color=colors)
        ax.set_yticks(range(len(names))[::-1])
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("test macro-F1")
        ax.set_title("3-class" if protocol == "3class" else "binary (stress vs baseline)")
        ax.grid(axis="x", alpha=0.3)
        for i, v in zip(range(len(names))[::-1], vals):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def per_class_chart(details, out: Path) -> None:
    usable = {k: v for k, v in details.items() if v}
    if not usable:
        return
    classes = sorted({c for v in usable.values() for c in v})
    models = sorted(usable)
    x = np.arange(len(classes))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * len(models), 4.4), dpi=150)
    for i, m in enumerate(models):
        vals = [usable[m].get(c, {}).get("f1", 0.0) for c in classes]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(classes)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 (test)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def training_curves(out: Path) -> None:
    series = []
    for path in sorted(RESULTS_DIR.glob("*_metrics.json")):
        data = load_json(path)
        if not data or "history" not in data:
            continue
        hist = data["history"]
        if hist.get("val_macro_f1"):
            series.append((data.get("tag", path.stem), hist))
    if not series:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=150)
    for name, hist in series:
        axes[0].plot(hist["val_macro_f1"], label=name, lw=1.6)
        axes[1].plot(hist["train_loss"], lw=1.6, linestyle="--", label=f"{name} train")
        axes[1].plot(hist["val_loss"], lw=1.6, label=f"{name} val")
    axes[0].set_title("Validation macro-F1")
    axes[1].set_title("Loss")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] wrote {out}")


def main() -> int:
    protocols, details = collect()
    if not any(protocols.values()):
        print("[plot] no metrics found in results/. Train first: python scripts/train.py")
        return 1

    md = markdown_table(protocols)
    (RESULTS_DIR / "model_comparison.md").write_text(md)
    print("[plot] wrote results/model_comparison.md")
    print("\n" + md)

    bar_chart(protocols, RESULTS_DIR / "model_comparison.png")
    per_class_chart(details, RESULTS_DIR / "per_class_f1.png")
    training_curves(RESULTS_DIR / "training_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
