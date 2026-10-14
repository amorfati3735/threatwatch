"""
Live inference for the physiological sequence model.

The predecessor implementation had two problems this fixes:

1. it hard-coded a temperature of ``2.2`` with no calibration; here the
   temperature is read from the checkpoint, where it was *fitted* on the
   validation split;
2. it did not know which channels or window length it had been trained with, so
   a mismatch produced silently wrong numbers; here the checkpoint carries the
   channel list, the window configuration and the standardizer, and a mismatch
   is a loud error.

The ``SensorMLPredictor.stress_index`` is the single number the fusion engine
consumes: the calibrated probability of the stress class (or, for a binary
model, the probability of the non-baseline class).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from threatwatch.config import CLASS_NAMES, MODELS_DIR, MODELS_DIR as _MODELS_DIR
from threatwatch.models.seq_models import build_model

DEFAULT_CHECKPOINT = MODELS_DIR / "cnn_transformer.pt"


class SensorMLPredictor:
    """Loads a checkpoint and classifies incoming windows with temporal smoothing."""

    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        device: str = "auto",
        smooth: int = 5,
        scaler_path: Optional[Path] = None,
        window_override: Optional[Dict] = None,
    ):
        self.checkpoint_path = Path(checkpoint)
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.smooth = max(1, int(smooth))

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Train one first:  python scripts/train.py --models cnn_transformer"
            )

        # ``weights_only=False`` is required because the checkpoint carries plain
        # python metadata (channel names, window dict, scaler stats).
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        self.model_name: str = ckpt["model_name"]
        self.model_kwargs: Dict = ckpt.get("model_kwargs") or {}
        self.n_channels: int = int(ckpt["n_channels"])
        self.class_names: List[str] = list(ckpt.get("class_names", CLASS_NAMES))
        n_classes = int(ckpt.get("n_classes", len(self.class_names)))
        self.temperature: float = float(ckpt.get("temperature", 1.0) or 1.0)

        self.channels: Optional[List[str]] = list(ckpt["channels"]) if ckpt.get("channels") else None

        # A checkpoint MUST describe the window it was trained with.  Guessing a
        # default here is exactly the silent-mismatch bug this project set out to
        # remove: the network would happily consume a window of the wrong length
        # and return confident nonsense.
        window = ckpt.get("window")
        if not window:
            if window_override is None:
                raise ValueError(
                    f"{self.checkpoint_path.name} does not record the window configuration it "
                    f"was trained with (it predates the self-describing checkpoint format). "
                    f"Retrain it (scripts/train.py), or pass "
                    f"window_override={{'window_sec': .., 'target_fs': ..}} if you know the value. "
                    f"Refusing to guess, because the wrong window silently corrupts inference."
                )
            window = window_override
        self.window: Dict = dict(window)
        self.window_sec: float = float(self.window["window_sec"])
        self.target_fs: float = float(self.window["target_fs"])

        self.model = build_model(self.model_name, self.n_channels, n_classes, **self.model_kwargs)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        # ---- standardizer -------------------------------------------------- #
        scaler = ckpt.get("scaler")
        if scaler is None and scaler_path is not None and Path(scaler_path).exists():
            scaler = json.loads(Path(scaler_path).read_text())
        if scaler is None:
            sidecar = _MODELS_DIR / "scaler.json"
            if sidecar.exists():
                scaler = json.loads(sidecar.read_text())

        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        # "global" uses the stored training statistics; "window" z-scores the
        # window against itself and needs no stored statistics at all.  The mode
        # is read from the checkpoint so live inference cannot normalise
        # differently from training.
        self.normalize_mode: str = str((scaler or {}).get("mode", "global"))
        if scaler is not None and scaler.get("mean") is not None:
            self.mean = np.asarray(scaler["mean"], dtype=np.float32)
            self.std = np.asarray(scaler["std"], dtype=np.float32)
            if self.channels is None:
                self.channels = list(scaler.get("channels", [])) or None
        if self.normalize_mode == "window" and self.channels is None and scaler is not None:
            self.channels = list(scaler.get("channels", [])) or None

        self._hist: Deque[np.ndarray] = deque(maxlen=self.smooth)

    # -- metadata ---------------------------------------------------------- #
    @property
    def window_samples(self) -> int:
        return int(round(self.window_sec * self.target_fs))

    @property
    def stress_class_index(self) -> int:
        """Index of the class whose probability is used as the distress signal."""
        names = [n.lower() for n in self.class_names]
        for key in ("stress", "distress", "threat"):
            if key in names:
                return names.index(key)
        return 1 if len(names) > 1 else 0

    def describe(self) -> Dict:
        return {
            "checkpoint": str(self.checkpoint_path),
            "model": self.model_name,
            "channels": self.channels,
            "window_sec": self.window_sec,
            "target_fs": self.target_fs,
            "window_samples": self.window_samples,
            "class_names": self.class_names,
            "temperature": self.temperature,
            "normalize": self.normalize_mode,
            "device": str(self.device),
        }

    # -- preprocessing ----------------------------------------------------- #
    def preprocess(self, window: np.ndarray) -> np.ndarray:
        """Apply the checkpoint's training-time normalisation to a raw ``(T, C)`` window."""
        w = np.nan_to_num(np.asarray(window, dtype=np.float32), nan=0.0)
        if self.normalize_mode == "window":
            mu = w.mean(axis=0, keepdims=True)
            sd = w.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-6, 1.0, sd)
            return ((w - mu) / sd).astype(np.float32)
        if self.mean is not None:
            if self.mean.shape[-1] != w.shape[-1]:
                raise ValueError(
                    f"Channel mismatch: checkpoint was trained on {self.mean.shape[-1]} channels "
                    f"but the window has {w.shape[-1]}. Channels: {self.channels}. "
                    f"Use the matching preprocessed dataset / scaler."
                )
            w = (w - self.mean) / self.std
        return w.astype(np.float32)

    # -- inference --------------------------------------------------------- #
    @torch.no_grad()
    def predict_proba(self, window: np.ndarray) -> np.ndarray:
        w = self.preprocess(window)
        x = torch.as_tensor(w[None], dtype=torch.float32).to(self.device)
        logits = self.model(x) / self.temperature
        return F.softmax(logits, dim=-1)[0].cpu().numpy()

    def predict(self, window: np.ndarray) -> Tuple[int, np.ndarray]:
        """Return ``(smoothed class index, smoothed probability vector)``."""
        probs = self.predict_proba(window)
        self._hist.append(probs)
        smooth = np.mean(np.stack(self._hist), axis=0)
        smooth = smooth / max(1e-9, smooth.sum())
        return int(np.argmax(smooth)), smooth

    def stress_index(self, probs: Sequence[float]) -> float:
        """The calibrated distress signal consumed by the fusion engine."""
        p = np.asarray(probs, dtype=np.float64)
        if len(p) == 2:
            return float(p[1])          # binary: non-baseline
        idx = self.stress_class_index
        return float(p[idx]) if idx < len(p) else 0.0

    def reset(self) -> None:
        self._hist.clear()


# Backwards-compatible alias for scripts written against the old API.
LivePredictor = SensorMLPredictor


def format_prediction(class_names: Sequence[str], cls: int, probs: np.ndarray) -> str:
    labels = " | ".join(f"{n}:{p * 100:4.1f}%" for n, p in zip(class_names, probs))
    name = class_names[cls] if 0 <= cls < len(class_names) else str(cls)
    return f"STATE = {name.upper():9s}  [{labels}]"


def load_latest_checkpoint(models_dir: Path = MODELS_DIR) -> Optional[Path]:
    """Pick the most recently modified checkpoint, preferring cnn_transformer."""
    candidates = sorted(models_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    for p in candidates:
        if "cnn_transformer" in p.name:
            return p
    return candidates[-1]
