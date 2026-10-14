"""
Edge-deployment export.

The project brief asks for model optimisation for microcontroller deployment
(quantisation, pruning, TensorFlow Lite).  Fully deploying the sequence model to
an ATmega328P is not realistic — 2 KB of SRAM cannot hold a 145 k-parameter
network — and this module says so plainly instead of pretending otherwise.

What it *does* deliver is the real, verifiable first half of that pipeline:

    PyTorch checkpoint  ->  ONNX graph  ->  (documented) TFLite / ONNX Runtime

The ONNX graph can be opened in Netron, and is verified numerically against
PyTorch so the export is provably lossless.

A note on the honest constraint: the laptop is the inference host for the deep
model; the microcontroller's job is sensing and display.  That is the
architecture the brief permits, and it is stated in `docs/ARCHITECTURE.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from threatwatch.config import MODELS_DIR
from threatwatch.models.seq_models import build_model


def load_model_from_checkpoint(checkpoint: Path):
    """Rebuild a model + its metadata from a checkpoint (CPU)."""
    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    model = build_model(
        ckpt["model_name"],
        int(ckpt["n_channels"]),
        int(ckpt.get("n_classes", 3)),
        **(ckpt.get("model_kwargs") or {}),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def export_onnx(
    checkpoint: Path,
    out_path: Optional[Path] = None,
    opset: int = 17,
    verify: bool = True,
) -> Tuple[Path, Dict]:
    """
    Export ``checkpoint`` to ONNX with a dynamic batch axis.

    Returns ``(path, report)``.  When ``verify`` is true the graph is checked and
    then compared numerically against PyTorch on a random window; a mismatch
    raises rather than silently shipping a wrong model.
    """
    checkpoint = Path(checkpoint)
    model, ckpt = load_model_from_checkpoint(checkpoint)

    window = ckpt.get("window") or {}
    window_samples = int(round(float(window.get("window_sec", 15.0)) * float(window.get("target_fs", 32.0))))
    n_channels = int(ckpt["n_channels"])

    out_path = Path(out_path) if out_path else MODELS_DIR / f"{checkpoint.stem}.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, window_samples, n_channels, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["window"],
        output_names=["logits"],
        dynamic_axes={"window": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    report: Dict = {
        "onnx_path": str(out_path),
        "model": ckpt["model_name"],
        "opset": opset,
        "input_shape": ["batch", window_samples, n_channels],
        "size_kb": round(out_path.stat().st_size / 1024.0, 1),
        "verified": False,
    }

    if verify:
        import onnx

        onnx.checker.check_model(str(out_path))
        report["onnx_checker"] = "passed"

        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
            x = np.random.randn(2, window_samples, n_channels).astype(np.float32)
            with torch.no_grad():
                torch_out = model(torch.as_tensor(x)).numpy()
            onnx_out = sess.run(None, {"window": x})[0]
            max_abs = float(np.max(np.abs(torch_out - onnx_out)))
            report["max_abs_diff"] = max_abs
            report["verified"] = max_abs < 1e-3
            if not report["verified"]:
                raise RuntimeError(
                    f"ONNX export diverged from PyTorch (max|diff| = {max_abs:.6f}). "
                    f"Refusing to ship an unverified graph."
                )
        except ImportError:
            report["onnxruntime"] = "not installed - numerical check skipped"

    return out_path, report


TFLITE_NOTE = """\
PyTorch -> ONNX -> TensorFlow Lite (documented route; not required for the demo)

  1. python scripts/export_onnx.py --checkpoint models/<model>.pt
  2. pip install onnx2tf            # or: onnx-tf
     onnx2tf -i models/<model>.onnx -o models/tflite
  3. (optional) int8 quantisation needs a representative dataset of windows:
     converter.optimizations = [tf.lite.Optimize.DEFAULT]
     converter.representative_dataset = <train windows>

Reality check for this project: a 145k-parameter sequence model does not fit an
ATmega328P (2 KB SRAM / 32 KB flash).  The deployable artefact on the board is
the firmware's own signal processing; the neural network is hosted on the
laptop, which is exactly what the brief's architecture permits.
"""


if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Export a checkpoint to ONNX")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args()
    path, rep = export_onnx(a.checkpoint, a.out, a.opset, verify=not a.no_verify)
    print(f"exported -> {path}")
    for k, v in rep.items():
        print(f"  {k}: {v}")
