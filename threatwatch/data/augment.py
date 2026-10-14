"""
Window augmentation for training.

Why this exists: under a subject-independent split the held-out subjects are
*not* drawn from the same distribution — a subject's resting heart rate, skin
temperature and EDA baseline are personal offsets that no amount of training data
from other people teaches the model.  Without augmentation the sequence models
memorise the ten training subjects and then generalise badly (this was measured:
the 1-D CNN reached val macro-F1 0.52 but test macro-F1 0.32).

These transforms deliberately model the *nuisances* that vary between people and
between wear sessions, rather than adding arbitrary noise:

* ``jitter``          — sensor noise floor
* ``scale``           — per-channel gain/offset drift, different skin contact
* ``shift``           — where in the window the event falls
* ``time_mask``       — a short dropout (packet loss / motion artefact)
* ``channel_dropout`` — a sensor briefly failing to report

All transforms operate on an already-standardised window, so unit expectations are
comparable across channels.  Nothing here looks at labels or at any other window,
so augmentation cannot leak information.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class AugmentConfig:
    """Knobs for :func:`augment_window`.  All default to `off`-able magnitudes."""

    jitter_std: float = 0.05
    scale_std: float = 0.10
    max_shift_frac: float = 0.10
    time_mask_frac: float = 0.125
    time_mask_p: float = 0.30
    channel_dropout_p: float = 0.20
    enabled: bool = True


DEFAULT_AUGMENT = AugmentConfig()


def augment_window(x, cfg: AugmentConfig = DEFAULT_AUGMENT, generator=None):
    """
    Augment one standardised window of shape ``(T, C)``.

    Imported lazily by :class:`~threatwatch.data.dataset.WindowDataset` as a torch
    tensor transform.  Kept dependency-light so it can also run on numpy arrays in
    a preprocessing script.
    """
    import torch

    if not cfg.enabled:
        return x

    g = generator
    T, C = x.shape
    out = x.clone()

    # 1. sensor jitter
    if cfg.jitter_std > 0:
        out = out + torch.randn(out.shape, generator=g, dtype=out.dtype) * cfg.jitter_std

    # 2. per-channel gain and offset (different skin contact / calibration)
    if cfg.scale_std > 0:
        gain = 1.0 + torch.randn(C, generator=g, dtype=out.dtype) * cfg.scale_std
        offset = torch.randn(C, generator=g, dtype=out.dtype) * cfg.scale_std
        out = out * gain + offset

    # 3. circular time shift — where in the window the event sits
    if cfg.max_shift_frac > 0:
        max_shift = int(T * cfg.max_shift_frac)
        if max_shift > 0:
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,), generator=g).item())
            if shift:
                out = torch.roll(out, shifts=shift, dims=0)

    # 4. short time mask (packet loss / motion artefact)
    if cfg.time_mask_p > 0 and torch.rand(1, generator=g).item() < cfg.time_mask_p:
        span = max(1, int(T * cfg.time_mask_frac))
        start = int(torch.randint(0, max(1, T - span), (1,), generator=g).item())
        out[start: start + span] = 0.0

    # 5. one channel briefly stops reporting
    if cfg.channel_dropout_p > 0 and torch.rand(1, generator=g).item() < cfg.channel_dropout_p:
        ch = int(torch.randint(0, C, (1,), generator=g).item())
        out[:, ch] = 0.0

    return out
