"""Classification metrics + plots used by all models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from threatwatch.config import CLASS_NAMES


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Sequence[str] = CLASS_NAMES
) -> Dict:
    """Return a JSON-serializable metric dict (overall + per-class)."""
    labels = list(range(len(class_names)))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        # Raw predictions are kept so a caller (e.g. the k-fold protocol) can
        # aggregate them without re-running the model.
        "y_true": [int(v) for v in np.asarray(y_true).reshape(-1)],
        "y_pred": [int(v) for v in np.asarray(y_pred).reshape(-1)],
        "per_class": {
            name: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm,
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=list(class_names), zero_division=0
        ),
    }


def save_confusion_matrix(
    cm: Sequence[Sequence[int]],
    path: Path,
    class_names: Sequence[str] = CLASS_NAMES,
    title: str = "Confusion matrix",
    normalize: bool = False,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_arr = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True)
        cm_arr = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm_arr.max() / 2 if cm_arr.max() > 0 else 0.5
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            txt = f"{cm_arr[i, j]:.2f}" if normalize else f"{int(cm_arr[i, j])}"
            ax.text(
                j, i, txt, ha="center", va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def summarize(metrics: Dict, name: str = "") -> str:
    lines = [f"=== {name} ===" if name else "==="]
    lines.append(f"accuracy : {metrics['accuracy']:.4f}")
    lines.append(f"macro-F1 : {metrics['macro_f1']:.4f}")
    lines.append(f"weight-F1: {metrics['weighted_f1']:.4f}")
    for cls, m in metrics["per_class"].items():
        lines.append(
            f"  {cls:>10}: P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} n={m['support']}"
        )
    return "\n".join(lines)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
