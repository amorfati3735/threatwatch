"""
Training / evaluation engine for the deep sequence models.

Improvements over the predecessor implementation
------------------------------------------------
* **LR warmup + cosine decay** — the previous run used a flat 1e-3 with cosine
  decay only, which is why the CNN peaked at epoch 1 and the BiLSTM diverged.
* **Gradient clipping, label smoothing and class weighting** are all explicit
  and configurable.
* **TensorBoard** logging (loss, val macro-F1, per-class F1, confusion matrix).
* **Self-describing checkpoints** — a checkpoint carries the channel list, the
  window configuration, the standardizer statistics and the fitted temperature,
  so live inference can never silently mismatch what training saw.
* **Temperature calibration** is fitted on the validation split and stored, so
  the live probabilities are honest rather than arbitrarily scaled by 2.2.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from threatwatch.config import (
    CLASS_NAMES,
    MODELS_DIR,
    RESULTS_DIR,
    TrainConfig,
)
from threatwatch.data.dataset import WindowDataset
from threatwatch.ml.metrics import compute_metrics, save_confusion_matrix
from threatwatch.models.seq_models import build_model, count_parameters


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """Linear warmup followed by cosine decay, stepped per batch."""
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, int(total_steps * cfg.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        preds.append(model(xb).argmax(dim=-1).cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(targets)


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return raw logits and targets for the whole loader (used for calibration)."""
    model.eval()
    logits, targets = [], []
    for xb, yb in loader:
        logits.append(model(xb.to(device)).cpu())
        targets.append(yb)
    return torch.cat(logits), torch.cat(targets)


@torch.no_grad()
def _measure_latency(model: nn.Module, x: np.ndarray, device: torch.device, n: int = 50) -> float:
    model.eval()
    xb = torch.as_tensor(x, dtype=torch.float32).to(device)
    for _ in range(5):  # warmup
        model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(xb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


# --------------------------------------------------------------------------- #
# TensorBoard (fail-soft: training must never depend on the logger)
# --------------------------------------------------------------------------- #
class _TBLogger:
    def __init__(self, tag: str, enabled: bool):
        self.writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = RESULTS_DIR / "tensorboard" / tag
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
        except Exception as exc:  # pragma: no cover
            print(f"[tb] disabled ({exc})")

    def scalar(self, name: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def text(self, name: str, value: str, step: int) -> None:
        if self.writer is not None:
            self.writer.add_text(name, value, step)

    def confusion(self, cm: Sequence[Sequence[int]], class_names: Sequence[str], path: Path, step: int) -> None:
        if self.writer is None:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cm_arr = np.asarray(cm, dtype=np.float64)
            row = cm_arr.sum(axis=1, keepdims=True)
            norm = np.divide(cm_arr, row, out=np.zeros_like(cm_arr), where=row != 0)
            fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=110)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=30, ha="right")
            ax.set_yticklabels(class_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                            color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
            fig.tight_layout()
            self.writer.add_figure("test/confusion_matrix", fig, step)
            plt.close(fig)
        except Exception:
            pass

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 200) -> float:
    """
    Fit a single temperature T minimising validation NLL (Guo et al., 2017).

    The predecessor project hard-coded ``T = 2.2`` with no justification; here T
    is measured and stored with the checkpoint.
    """
    log_T = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / torch.exp(log_T), targets)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        t = float(torch.exp(log_T).item())
        return float(np.clip(t, 0.05, 20.0))
    except Exception:
        return 1.0


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Standard 10-bin ECE — a number to report alongside accuracy."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / len(conf)) * abs(acc - avg_conf)
    return float(ece)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def train_deep(
    model_name: str,
    train: Tuple[np.ndarray, np.ndarray],
    val: Tuple[np.ndarray, np.ndarray],
    test: Tuple[np.ndarray, np.ndarray],
    tcfg: Optional[TrainConfig] = None,
    model_kwargs: Optional[dict] = None,
    tag: Optional[str] = None,
    channels: Optional[Sequence[str]] = None,
    window: Optional[dict] = None,
    scaler: Optional[dict] = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict:
    """
    Train one deep model and return a full results dict.

    ``train``/``val``/``test`` are ``(X, y)`` tuples with ``X`` of shape
    ``(N, T, C)``.  ``channels``, ``window`` and ``scaler`` are stored inside the
    checkpoint so the live path is self-describing.
    """
    tcfg = tcfg or TrainConfig()
    class_names = list(class_names)
    n_classes = len(class_names)

    torch.manual_seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    X_tr, y_tr = train
    X_va, y_va = val
    X_te, y_te = test

    if tcfg.max_train_windows and len(X_tr) > tcfg.max_train_windows:
        idx = np.random.RandomState(tcfg.seed).choice(len(X_tr), tcfg.max_train_windows, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    n_channels = X_tr.shape[-1]
    train_loader = DataLoader(WindowDataset(X_tr, y_tr, augment=tcfg.augment),
                              batch_size=tcfg.batch_size,
                              shuffle=True, num_workers=tcfg.num_workers, drop_last=False)
    val_loader = DataLoader(WindowDataset(X_va, y_va), batch_size=256,
                            shuffle=False, num_workers=tcfg.num_workers)
    test_loader = DataLoader(WindowDataset(X_te, y_te), batch_size=256,
                             shuffle=False, num_workers=tcfg.num_workers)

    device = get_device(tcfg)
    model = build_model(model_name, n_channels, n_classes, **(model_kwargs or {})).to(device)
    n_params = count_parameters(model)

    weight = _class_weights(y_tr, n_classes).to(device) if tcfg.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=tcfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = _build_scheduler(optimizer, tcfg, len(train_loader))

    tag = tag or model_name
    tb = _TBLogger(tag, tcfg.tensorboard)

    print(f"[train] model={model_name} params={n_params:,} device={device} "
          f"train={len(X_tr)} val={len(X_va)} test={len(X_te)} "
          f"window={X_tr.shape[1]}x{X_tr.shape[2]} bs={tcfg.batch_size} "
          f"augment={tcfg.augment} normalize={tcfg.normalize}")

    best_f1, best_state, best_epoch = -1.0, None, -1
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "lr": []}
    t0 = time.time()

    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        running, lr_now = 0.0, optimizer.param_groups[0]["lr"]
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(yb)
        train_loss = running / max(1, len(X_tr))

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vloss += criterion(model(xb), yb).item() * len(yb)
        vloss /= max(1, len(X_va))
        vp, vt = _predict(model, val_loader, device)
        vm = compute_metrics(vt, vp, class_names)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(vloss)
        history["val_macro_f1"].append(vm["macro_f1"])
        history["lr"].append(lr_now)
        tb.scalar("loss/train", train_loss, epoch)
        tb.scalar("loss/val", vloss, epoch)
        tb.scalar("metrics/val_macro_f1", vm["macro_f1"], epoch)
        tb.scalar("metrics/val_accuracy", vm["accuracy"], epoch)
        tb.scalar("lr", lr_now, epoch)

        improved = vm["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch = vm["macro_f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"  epoch {epoch:02d}/{tcfg.epochs} | train_loss={train_loss:.4f} "
              f"val_loss={vloss:.4f} val_macroF1={vm['macro_f1']:.4f} "
              f"lr={lr_now:.2e}{' *' if improved else ''}")

        if epoch - best_epoch >= tcfg.patience:
            print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- temperature calibration on the validation split ------------------- #
    val_logits, val_targets = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_targets)
    probs_uncal = F.softmax(val_logits, dim=-1).numpy()
    probs_cal = F.softmax(val_logits / temperature, dim=-1).numpy()
    ece_before = expected_calibration_error(probs_uncal, val_targets.numpy())
    ece_after = expected_calibration_error(probs_cal, val_targets.numpy())
    print(f"[train] temperature={temperature:.3f}  ECE {ece_before:.4f} -> {ece_after:.4f}")

    val_pred, val_true = _predict(model, val_loader, device)
    test_pred, test_true = _predict(model, test_loader, device)
    val_metrics = compute_metrics(val_true, val_pred, class_names)
    test_metrics = compute_metrics(test_true, test_pred, class_names)

    # ---- persist ----------------------------------------------------------- #
    ckpt_path = MODELS_DIR / f"{tag}.pt"
    checkpoint = {
        "model_name": model_name,
        "model_kwargs": model_kwargs or {},
        "n_channels": n_channels,
        "n_classes": n_classes,
        "class_names": class_names,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "best_epoch": best_epoch,
        "val_macro_f1": best_f1,
        "temperature": temperature,
        "channels": list(channels) if channels is not None else None,
        "window": window or None,
        "scaler": scaler or None,
    }
    torch.save(checkpoint, ckpt_path)

    latency_ms = _measure_latency(model, X_te[:1], device)

    results = {
        "model": model_name,
        "model_kwargs": model_kwargs or {},
        "params": n_params,
        "tag": tag,
        "best_epoch": best_epoch,
        "train_time_sec": round(train_time, 2),
        "inference_latency_ms": round(latency_ms, 3),
        "temperature": round(temperature, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "window_shape": list(X_te.shape[1:]),
        "channels": list(channels) if channels is not None else None,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{tag}_metrics.json").write_text(json.dumps(results, indent=2))
    save_confusion_matrix(test_metrics["confusion_matrix"], RESULTS_DIR / f"{tag}_confusion.png",
                          class_names, title=f"{model_name} — test", normalize=True)
    tb.confusion(test_metrics["confusion_matrix"], class_names,
                 RESULTS_DIR / f"{tag}_confusion.png", best_epoch)
    tb.close()

    print(f"[train] saved checkpoint -> {ckpt_path}")
    print(f"[train] test macro-F1={test_metrics['macro_f1']:.4f} "
          f"acc={test_metrics['accuracy']:.4f} latency={latency_ms:.2f} ms")
    return results
