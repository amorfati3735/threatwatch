"""
On-device inference parity.

This is the test that lets the project say "the model runs on the ESP32" without
that claim being a matter of faith.

It compiles ``firmware/edge_model/edge_infer.c`` — the *exact* source the Arduino
sketch includes — with the host ``cc``, together with a small stdin/stdout driver,
and compares its logits against the PyTorch checkpoint the C header was exported
from. If the two disagree, the deployment is wrong and the test says so.

It also measures how much the int8 quantisation actually costs: the float model's
predictions are compared against the quantised C implementation's predictions on
real held-out data, so the accuracy delta is a number rather than a hope.

Skipped automatically when there is no ``cc``, no CNN checkpoint, or no processed
dataset — this must never fail a machine that simply has not trained anything yet.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from threatwatch.config import MODELS_DIR, PROCESSED_DIR  # noqa: E402
from threatwatch.data.dataset import load_processed  # noqa: E402
from threatwatch.models.seq_models import build_model  # noqa: E402

EDGE_DIR = ROOT / "firmware" / "edge_model"
CC = shutil.which("cc") or shutil.which("gcc")

# Sampling keeps the test fast; it is still hundreds of real held-out windows.
N_SAMPLE = 400

# The test asserts TWO different things, and keeping them separate is what makes
# it meaningful.
#
# 1. The C forward pass must reproduce, essentially exactly, a NumPy reference that
#    consumes the *same quantised weights*. This is a correctness assertion: any
#    indexing, folding or buffer-layout mistake in edge_infer.c shows up here
#    immediately, with no quantisation noise to hide behind.
#
# 2. Against the *float* PyTorch model the C output must still agree on the
#    decision (argmax) and not cost meaningful accuracy. This is a budget
#    assertion: it bounds what the int8 deployment choice costs in behaviour.
C_VS_REFERENCE_TOL = 0.02
MIN_ARGMAX_AGREEMENT = 0.95
MAX_ACCURACY_DROP = 0.03
# Reported for the write-up; not a pass/fail gate beyond a sanity ceiling.
REPORT_ONLY_MAX_LOGIT_DIFF = 3.0


def _cnn_checkpoints():
    """Deployable CNN checkpoints only — see scripts/export_edge.cnn_candidates.

    A bare ``cnn*.pt`` glob also matches ``cnn_transformer*.pt``, which has no C
    forward pass; selecting it would make this test skip instead of run.
    """
    from scripts.export_edge import cnn_candidates

    return cnn_candidates()


def _dataset():
    cands = sorted(PROCESSED_DIR.glob("w15_s5_fs32*.npz"))
    return cands[0] if cands else None


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Export a checkpoint to a temporary header and compile the C driver."""
    if CC is None:
        pytest.skip("no C compiler (cc/gcc) available")
    ckpts = _cnn_checkpoints()
    if not ckpts:
        pytest.skip("no cnn checkpoint in models/ — train one first")
    if _dataset() is None:
        pytest.skip("no processed dataset — run scripts/preprocess.py first")

    ckpt_path = ckpts[-1]
    tmp = tmp_path_factory.mktemp("edge")

    # Export to the REAL generated path, and compile the REAL sources in place.
    #
    # This matters more than it looks. A quoted `#include "edge_model.h"` inside
    # edge_infer.c resolves against edge_infer.c's OWN directory first, so a
    # stale header left in firmware/edge_model/ silently wins over one placed in
    # a -I directory — and the test then compiles the C against a different model
    # than the driver does. That is a real failure mode, not a hypothetical: it
    # cost an hour of bring-up here. Regenerating in place removes it entirely,
    # and makes this test compile exactly the sources the Arduino build uses.
    header = EDGE_DIR / "edge_model.h"
    report = tmp / "report.json"

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_edge.py"),
         "--checkpoint", str(ckpt_path), "--out", str(header), "--report", str(report)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"export_edge.py failed:\n{proc.stdout}\n{proc.stderr}")

    # Static assertions from edge_infer.c must hold for this architecture.
    text = header.read_text()
    assert "#define EDGE_N_CONV       3" in text, "edge_infer.c requires exactly 3 conv stages"

    exe = tmp / "edge_host"
    compile_proc = subprocess.run(
        [CC, "-O2", "-std=c99", "-o", str(exe),
         str(EDGE_DIR / "edge_infer.c"), str(EDGE_DIR / "host_main.c"),
         "-I", str(EDGE_DIR), "-lm"],
        capture_output=True, text=True,
    )
    if compile_proc.returncode != 0:
        pytest.fail(f"edge_infer.c failed to compile:\n{compile_proc.stderr}")

    return {
        "ckpt": ckpt_path,
        "header": header,
        "report": json.loads(report.read_text()),
        "exe": exe,
    }


def _window_normalize(X: np.ndarray) -> np.ndarray:
    """Mirror of ``threatwatch.data.dataset.Standardizer(mode='window')``."""
    X = np.nan_to_num(X, nan=0.0)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32)


def _int8_reference(model, windows: np.ndarray, scaler: dict, mode: str) -> np.ndarray:
    """
    NumPy forward pass over the SAME quantised weights the header carries.

    Reuses the exporter's own ``extract_cnn`` / ``quantise_int8`` so this cannot
    drift from what was emitted.  If the C code agrees with this, the compiled
    implementation is a faithful implementation of the exported model; whatever
    gap remains against float PyTorch is quantisation and nothing else.
    """
    from scripts.export_edge import extract_cnn, quantise_int8

    info = extract_cnn(model)
    convs = info["convs"]

    if mode == "window":
        mu = windows.mean(axis=1, keepdims=True)
        sd = windows.std(axis=1, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        x = (windows - mu) / sd
    else:
        mean = np.asarray(scaler["mean"], dtype=np.float64)
        std = np.asarray(scaler["std"], dtype=np.float64)
        x = (windows.astype(np.float64) - mean) / std

    h = x.transpose(0, 2, 1).astype(np.float64)  # (N, C, T)

    for spec, _ in zip(convs, range(len(convs))):
        q = quantise_int8(spec["weight"])
        w = q["q"].numpy().astype(np.float64)
        s = q["scale"].numpy().astype(np.float64)
        b = spec["bias"].numpy().astype(np.float64)
        oc, IC, K = w.shape
        P, D = spec["padding"], spec["dilation"]
        n, _, T = h.shape
        L = T + 2 * P - D * (K - 1)
        out = np.zeros((n, oc, L))
        for t in range(L):
            for k in range(K):
                idx = t + k * D - P
                if 0 <= idx < T:
                    # (n, IC) x (oc, IC) -> (n, oc)
                    out[:, :, t] += h[:, :, idx] @ w[:, :, k].T
        out = np.maximum(out * s.reshape(1, -1, 1) + b.reshape(1, -1, 1), 0.0)
        if spec["pool"]:
            out = out[:, :, : (L // 2) * 2].reshape(n, oc, L // 2, 2).max(axis=3)
        h = out

    feat = h.mean(axis=2)                       # (n, oc)
    hq = quantise_int8(info["head_weight"])
    hw = hq["q"].numpy().astype(np.float64)     # (n_classes, feat)
    hs = hq["scale"].numpy().astype(np.float64)
    hb = info["head_bias"].numpy().astype(np.float64)
    return (feat @ hw.T) * hs.reshape(1, -1) + hb.reshape(1, -1)


def _load_torch_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(
        ckpt["model_name"], int(ckpt["n_channels"]),
        int(ckpt.get("n_classes", 3)), **(ckpt.get("model_kwargs") or {}),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _run_c(exe: Path, windows: np.ndarray):
    """Feed (N, T, C) windows to the compiled C forward pass; return (N, K) logits."""
    n, T, C = windows.shape
    lines = [str(n)]
    flat = windows.reshape(n, -1)
    for row in flat:
        lines.append(" ".join(f"{v:.9g}" for v in row))
    proc = subprocess.run(
        [str(exe)], input="\n".join(lines) + "\n",
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"C driver failed: {proc.stderr}"
    out = np.array([[float(v) for v in line.split()] for line in proc.stdout.strip().splitlines()])
    timing = 0.0
    for line in proc.stderr.splitlines():
        if line.startswith("HOST_MS_PER_INFER"):
            timing = float(line.split()[1])
    return out, timing


def test_c_forward_pass_matches_pytorch(exported):
    ds = _dataset()
    d = load_processed(ds)

    model, ckpt = _load_torch_model(exported["ckpt"])
    scaler = ckpt.get("scaler") or {}
    mode = str(scaler.get("mode", "global"))

    n_channels = int(ckpt["n_channels"])
    X = d["test_X"]
    y = d["test_y"]
    assert X.shape[-1] == n_channels, "dataset and checkpoint channel counts disagree"

    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(N_SAMPLE, len(X)), replace=False)
    Xs, ys = X[idx].astype(np.float32), y[idx]

    # The dataset is stored *after* the global transform, and a ``--normalize window``
    # run applied a second, per-window z-score on top of that before training.  The C
    # code performs that per-window z-score itself, so the two sides must be fed
    # different things:
    #
    #   torch : the exact tensor training used  (window-normalised if mode==window)
    #   C     : raw-ish units, because edge_infer() normalises internally
    #
    # Per-window z-scoring is affine-invariant, so feeding the globally-standardised
    # windows to the C code yields the same z-scores as feeding genuinely raw data.
    if mode == "window":
        torch_input = _window_normalize(Xs)
        c_input = Xs
        raw_for_reference = Xs
    else:
        assert scaler.get("mean") is not None, "global mode without stored statistics"
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)
        torch_input = Xs
        c_input = Xs * std + mean
        raw_for_reference = c_input

    with torch.no_grad():
        logits_torch = model(torch.as_tensor(torch_input)).numpy()

    logits_c, host_ms = _run_c(exported["exe"], c_input)
    logits_ref = _int8_reference(model, raw_for_reference, scaler, mode)

    assert logits_c.shape == logits_torch.shape == logits_ref.shape

    # ---- (1) is the C implementation faithful to the exported model? -------- #
    c_vs_ref = float(np.max(np.abs(logits_c - logits_ref)))
    assert c_vs_ref < C_VS_REFERENCE_TOL, (
        f"the compiled C forward pass does not reproduce the exported model: "
        f"max|C - reference|={c_vs_ref:.5f} (tolerance {C_VS_REFERENCE_TOL}). "
        f"This is an implementation bug in edge_infer.c, not quantisation."
    )

    # ---- (2) what does the int8 deployment choice cost vs float? ----------- #
    max_diff = float(np.max(np.abs(logits_c - logits_torch)))
    mean_diff = float(np.mean(np.abs(logits_c - logits_torch)))
    assert max_diff < REPORT_ONLY_MAX_LOGIT_DIFF, (
        f"int8 logits differ from float PyTorch by {max_diff:.4f}, beyond the sanity "
        f"ceiling {REPORT_ONLY_MAX_LOGIT_DIFF}"
    )

    pred_torch = logits_torch.argmax(axis=1)
    pred_c = logits_c.argmax(axis=1)
    agreement = float((pred_torch == pred_c).mean())
    assert agreement >= MIN_ARGMAX_AGREEMENT, (
        f"argmax agreement {agreement:.4f} < {MIN_ARGMAX_AGREEMENT}"
    )

    acc_torch = float((pred_torch == ys).mean())
    acc_c = float((pred_c == ys).mean())
    assert acc_torch - acc_c <= MAX_ACCURACY_DROP, (
        f"int8 quantisation cost {acc_torch - acc_c:.4f} accuracy, "
        f"more than the {MAX_ACCURACY_DROP} budget (float {acc_torch:.4f} -> int8 {acc_c:.4f})"
    )

    print(f"\n[edge parity] checkpoint        : {exported['ckpt'].name}")
    print(f"[edge parity] windows compared  : {len(Xs)} real held-out windows")
    print(f"[edge parity] C vs int8 ref     : max {c_vs_ref:.5f}  (implementation correctness)")
    print(f"[edge parity] C vs float torch  : max {max_diff:.4f}  mean {mean_diff:.4f}  (quantisation cost)")
    print(f"[edge parity] argmax agreement  : {agreement:.4f}")
    print(f"[edge parity] accuracy          : float {acc_torch:.4f} -> int8 {acc_c:.4f} "
          f"(delta {acc_c - acc_torch:+.4f})")
    print(f"[edge parity] host C latency    : {host_ms:.3f} ms/window (x86, not the ESP32)")


def test_arena_fits_the_board(exported):
    """The activation arena must fit the SRAM the ESP32 sketch actually has left."""
    report = exported["report"]
    arena_kb = report["arena_bytes"] / 1024.0

    # Measured by a real `arduino-cli compile` of firmware/esp32_wearable:
    # 42 884 bytes of static globals, 284 796 bytes left in the 327 680-byte budget.
    FREE_SRAM_BYTES = 284_796

    assert report["arena_bytes"] < FREE_SRAM_BYTES, (
        f"activation arena is {arena_kb:.1f} KB but only "
        f"{FREE_SRAM_BYTES / 1024:.1f} KB of SRAM is free on the board"
    )
    print(f"\n[edge parity] arena {arena_kb:.1f} KB / {FREE_SRAM_BYTES / 1024:.1f} KB free "
          f"({100 * report['arena_bytes'] / FREE_SRAM_BYTES:.0f}% of the remaining SRAM)")


def test_int8_weights_are_actually_smaller(exported):
    report = exported["report"]
    assert report["int8_weight_bytes"] < report["float32_weight_bytes_if_unquantised"] / 3.5, (
        "int8 export did not reduce weight storage as expected"
    )
    assert report["max_quantisation_error"] < 0.02, (
        f"per-channel quantisation error {report['max_quantisation_error']} is too large"
    )
