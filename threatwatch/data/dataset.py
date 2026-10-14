"""
Windowing, standardization and dataset assembly.

Leakage rules enforced here
---------------------------
1. Windows are labelled by majority vote and windows dominated by
   transient/ignore segments are discarded.
2. The channel standardizer is fit on **training subjects only**.
3. Subjects are partitioned into train/val/test *before* any windowing, so a
   window from one subject can never appear in two splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from threatwatch.config import (
    FEATURE_CHANNELS,
    WindowConfig,
)
from threatwatch.data.wesad import extract_subject


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def extract_windows(
    subject_data: Dict[str, np.ndarray],
    window_samples: int,
    stride_samples: int,
    feature_channels: List[str] = FEATURE_CHANNELS,
    min_valid_frac: float = 0.75,
    min_majority: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window over a subject's synchronized signals.

    Returns
    -------
    X : (n_windows, window_samples, n_channels) float32
    y : (n_windows,) int64 class indices
    starts : (n_windows,) int64 start sample indices (for replay/real-time use)
    """
    feats = np.stack([subject_data[c] for c in feature_channels], axis=-1).astype(np.float32)
    labels = subject_data["label"]
    total = len(labels)

    xs, ys, starts = [], [], []
    for start in range(0, total - window_samples + 1, stride_samples):
        end = start + window_samples
        win = feats[start:end]
        win_lbl = labels[start:end]

        valid = win_lbl[win_lbl >= 0]
        if len(valid) < window_samples * min_valid_frac:
            continue
        counts = np.bincount(valid, minlength=3)
        majority = int(np.argmax(counts))
        # Require the majority class to dominate the *whole* window too.
        if counts[majority] < window_samples * min_majority:
            continue

        xs.append(win)
        ys.append(majority)
        starts.append(start)

    if not xs:
        return (
            np.empty((0, window_samples, len(feature_channels)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# Standardization
# --------------------------------------------------------------------------- #
@dataclass
class Standardizer:
    """
    Per-channel normalizer.

    Two modes, both leakage-free for different reasons:

    ``global``
        Subtract the mean / divide by the std computed over the **training
        subjects only**.  Absolute levels are preserved, which means each new
        subject's personal offset (resting HR, skin temperature, EDA baseline)
        shifts every channel at once and pushes windows off the training
        manifold.  This is what the predecessor project did, and it is a large
        part of why its subject-independent scores were poor.

    ``window``
        Z-score each window against *its own* per-channel mean and std.  This
        cancels the subject-specific offset by construction and needs no fitted
        statistics at all, so it is also exactly reproducible on the ESP32: the
        board normalises the window it just captured, with no dataset lookup.
        The trade-off is that it discards absolute level, so a slowly-rising
        baseline within a window is attenuated.

    The mode travels with the checkpoint, so the live path cannot apply a
    different normalisation from the one used in training.
    """

    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    channels: Tuple[str, ...] = tuple(FEATURE_CHANNELS)
    mode: str = "global"   # "global" | "window"

    def fit(self, X: np.ndarray) -> "Standardizer":
        # X: (N, T, C) -> compute stats over N and T
        self.mean = X.mean(axis=(0, 1)).astype(np.float32)
        self.std = X.std(axis=(0, 1)).astype(np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0)
        if self.mode == "window":
            mu = X.mean(axis=1, keepdims=True)
            sd = X.std(axis=1, keepdims=True)
            sd = np.where(sd < 1e-6, 1.0, sd)
            return ((X - mu) / sd).astype(np.float32)
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer not fitted")
        return ((X - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "channels": list(self.channels),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float32) if d.get("mean") is not None else None,
            std=np.asarray(d["std"], dtype=np.float32) if d.get("std") is not None else None,
            channels=tuple(d.get("channels", FEATURE_CHANNELS)),
            mode=str(d.get("mode", "global")),
        )


# --------------------------------------------------------------------------- #
# Dataset assembly / persistence
# --------------------------------------------------------------------------- #
def build_split(
    subjects: Iterable[str],
    wcfg: WindowConfig = WindowConfig(),
    data_dir: Optional[Path] = None,
    verbose: bool = True,
    channels: List[str] = FEATURE_CHANNELS,
) -> Dict[str, np.ndarray]:
    """Extract windows for a set of subjects (no cross-subject leakage)."""
    Xs, ys, subj_ids, starts = [], [], [], []
    for subj in subjects:
        sdata = extract_subject(subj, data_dir) if data_dir else extract_subject(subj)
        if sdata is None:
            if verbose:
                print(f"  [skip] {subj}: not found")
            continue
        X, y, st = extract_windows(
            sdata, wcfg.window_samples, wcfg.stride_samples, feature_channels=channels
        )
        if verbose:
            dist = np.bincount(y, minlength=3) if len(y) else np.zeros(3, int)
            print(
                f"  {subj}: {sdata['n']/wcfg.target_fs/60:5.1f} min -> "
                f"{len(y):5d} windows  [base={dist[0]} stress={dist[1]} amuse={dist[2]}]"
            )
        if len(y):
            Xs.append(X)
            ys.append(y)
            subj_ids.append(np.array([subj] * len(y)))
            starts.append(st)
    if not Xs:
        raise RuntimeError("No windows extracted — are the WESAD pickles present?")
    return {
        "X": np.concatenate(Xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "subjects": np.concatenate(subj_ids, axis=0),
        "starts": np.concatenate(starts, axis=0),
    }


def save_processed(
    path: Path,
    train: dict,
    val: dict,
    test: dict,
    wcfg: WindowConfig,
    channels: Iterable[str] = FEATURE_CHANNELS,
) -> None:
    """Persist a processed dataset.  The channel list travels with the tensors so a
    loaded dataset can never be silently mismatched against a model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        w_window_sec=wcfg.window_sec,
        w_stride_sec=wcfg.stride_sec,
        w_target_fs=wcfg.target_fs,
        channels=np.array(list(channels), dtype=object),
        train_X=train["X"], train_y=train["y"], train_subjects=train["subjects"],
        val_X=val["X"], val_y=val["y"], val_subjects=val["subjects"],
        test_X=test["X"], test_y=test["y"], test_subjects=test["subjects"],
    )


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #
class WindowDataset:
    """
    In-memory torch dataset over (N, T, C) windows (torch imported lazily).

    When ``augment`` is set each training sample is perturbed on the fly by
    :func:`threatwatch.data.augment.augment_window`, so the model sees a slightly
    different window every epoch without duplicating the tensor store.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        import torch

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.augment = augment
        self._rng = None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augment:
            from threatwatch.data.augment import augment_window

            if self._rng is None:
                import torch

                self._rng = torch.Generator().manual_seed(0)
            x = augment_window(x, generator=self._rng)
        return x, self.y[idx]


def load_processed(path: Path) -> dict:
    """Load a saved processed dataset (.npz)."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}
