#!/usr/bin/env python3
"""
Export a 1-D CNN checkpoint to a dependency-free, int8-quantised C header so the
network can run **on the ESP32 itself**.

    python scripts/export_edge.py                       # newest cnn checkpoint
    python scripts/export_edge.py --checkpoint models/cnn_edge.pt
    python scripts/export_edge.py --window-samples 480  # override the input length
    python scripts/export_edge.py --report results/edge_export.json

What it does
------------
1. folds every BatchNorm into its preceding convolution (so the on-device forward
   pass has one multiply-accumulate per tap and no separate normalisation stage);
2. quantises each convolution/linear weight to int8 with a **per-output-channel**
   scale (``max|W_out| / 127``);
3. writes ``firmware/edge_model/edge_model.h`` — weights, scales, float32 biases,
   the normalisation mode and its statistics, plus the exact activation-arena size
   the C code must allocate.

Activations stay float32.  That is the deliberate choice for this chip: the ESP32
has a hardware single-precision FPU, so keeping activations float avoids all
fixed-point requantisation error while still cutting weight storage ~4x.  For an
AVR target (2 KB SRAM) the whole approach is infeasible and the exporter says so.

The emitted C forward pass is verified against PyTorch by
``tests/test_edge_parity.py``, which compiles it on the host with ``cc``.
A model is not "deployed" because a file was copied to a board.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from threatwatch.config import BASE_DIR, CLASS_NAMES, MODELS_DIR, RESULTS_DIR
from threatwatch.models.seq_models import CNN1DClassifier

EDGE_DIR = BASE_DIR / "firmware" / "edge_model"
HEADER = EDGE_DIR / "edge_model.h"


# --------------------------------------------------------------------------- #
# Architecture extraction
# --------------------------------------------------------------------------- #
def cnn_candidates() -> List[Path]:
    """Checkpoints that are actually the deployable CNN, newest first.

    A ``cnn*.pt`` glob also matches ``cnn_transformer*.pt``; picking the newest
    match without checking the architecture silently selects a model with no C
    forward pass.  So the architecture recorded *inside* the checkpoint decides,
    not the filename.
    """
    out: List[Path] = []
    for path in sorted(MODELS_DIR.glob("cnn*.pt"), key=lambda q: q.stat().st_mtime):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if ck.get("model_name") == "cnn":
            out.append(path)
    return out


def extract_cnn(model: nn.Module) -> Dict:
    """
    Pull the convolution stack, pool positions and the classifier head out of a
    ``CNN1DClassifier``, folding BatchNorm as it goes.

    Reads the module tree rather than hard-coding layer indices, so changing the
    architecture (or its width) cannot silently export the wrong thing.
    """
    if not isinstance(model, CNN1DClassifier):
        raise TypeError(
            f"export_edge only supports the compact CNN (got {type(model).__name__}). "
            f"The attention/LSTM variants have no C forward pass; deploy the CNN."
        )

    convs: List[Dict] = []
    children = list(model.net.children())

    i = 0
    while i < len(children):
        layer = children[i]
        if isinstance(layer, nn.Conv1d):
            spec = {
                "in_ch": layer.in_channels,
                "out_ch": layer.out_channels,
                "kernel": layer.kernel_size[0],
                "stride": layer.stride[0],
                "padding": layer.padding[0],
                "dilation": layer.dilation[0],
                "weight": layer.weight.detach().clone(),
                "bias": None,
                "pool": 0,
                "group": layer.groups,
            }
            if spec["group"] != 1:
                raise NotImplementedError("grouped convolutions are not supported by edge_infer.c")
            if spec["stride"] != 1:
                raise NotImplementedError("stride != 1 is not supported by edge_infer.c")

            # nn.Conv1d defaults to bias=True.  That bias must survive the fold:
            #
            #   BN(conv(x) + cb) = scale * (conv(x) + cb) + beta - mean*scale
            #                    = conv(x)*scale + (cb*scale + beta - mean*scale)
            #
            # Dropping the `cb * scale` term is a silent, small, sign-preserving
            # error that only shows up as a parity failure — hence the test.
            conv_bias = (layer.bias.detach().clone() if layer.bias is not None
                         else torch.zeros(spec["out_ch"]))

            j = i + 1
            if j < len(children) and isinstance(children[j], nn.BatchNorm1d):
                bn = children[j]
                eps = bn.eps
                gamma = bn.weight.detach().clone()
                beta = bn.bias.detach().clone()
                mean = bn.running_mean.detach().clone()
                var = bn.running_var.detach().clone()
                scale = gamma / torch.sqrt(var + eps)
                spec["weight"] = spec["weight"] * scale.view(-1, 1, 1)
                spec["bias"] = beta - mean * scale + conv_bias * scale
                j += 1
            else:
                spec["bias"] = conv_bias

            # Skip the activation that sits between the BatchNorm and any pooling.
            # (The block order is Conv -> BN -> ReLU -> MaxPool, so the pool is not
            # adjacent to the BN.)  Getting this wrong silently doubles the
            # activation arena, which is the binding constraint on the board.
            k = j
            while k < len(children) and isinstance(children[k], (nn.ReLU, nn.GELU,
                                                                 nn.LeakyReLU, nn.SiLU)):
                k += 1
            if k < len(children) and isinstance(children[k], nn.MaxPool1d):
                pool = children[k]
                if pool.kernel_size != 2 or pool.stride not in (None, 2):
                    raise NotImplementedError("only MaxPool1d(2) with stride 2 is supported")
                spec["pool"] = int(pool.kernel_size)
                j = k + 1

            convs.append(spec)
            i = j
            continue
        i += 1

    if not convs:
        raise ValueError("no convolution layers found")

    # classifier head
    linear = None
    for m in model.head.modules():
        if isinstance(m, nn.Linear):
            linear = m
            break
    if linear is None:
        raise ValueError("no Linear layer found in the classifier head")

    return {
        "convs": convs,
        "n_channels": convs[0]["in_ch"],
        "n_classes": linear.out_features,
        "head_in": linear.in_features,
        "head_weight": linear.weight.detach().clone(),
        "head_bias": linear.bias.detach().clone(),
    }


def quantise_int8(w: torch.Tensor) -> Dict:
    """
    Per-output-channel symmetric int8 quantisation.

    ``w`` has shape ``(out, in, k)`` (or ``(out, in)`` for the dense layer).
    Returns the int8 array, the per-output scale and the reconstruction error.
    """
    out = w.shape[0]
    flat = w.reshape(out, -1)
    max_abs = flat.abs().amax(dim=1)
    scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
    q = torch.round(flat / scale.view(-1, 1)).clamp(-127, 127).to(torch.int8)
    deq = q.float() * scale.view(-1, 1)
    err = float((deq - flat).abs().max())
    return {"q": q.reshape(w.shape), "scale": scale, "max_err": err}


# --------------------------------------------------------------------------- #
# Activation arena
# --------------------------------------------------------------------------- #
def arena_size(convs: List[Dict], window: int) -> Dict:
    """
    Working memory for the C forward pass, in floats.

    The layout mirrors ``edge_infer.c``:

      * buffer A holds conv 0's full output, then its max-pooled result in place;
      * buffer B holds conv 1's full output, then its max-pooled result in place;
      * buffer C holds the last conv's output.

    Convolutions cannot run in place (output ``t`` reads neighbours), which is
    why A and B must each hold a full un-pooled tensor.
    """
    shapes = []
    length = window
    for spec in convs:
        L = (length + 2 * spec["padding"] - spec["dilation"] * (spec["kernel"] - 1) - 1) // spec["stride"] + 1
        shapes.append({"out_ch": spec["out_ch"], "len": L})
        length = L // spec["pool"] if spec["pool"] else L

    a = shapes[0]["out_ch"] * shapes[0]["len"]
    b = (shapes[1]["out_ch"] * shapes[1]["len"]) if len(shapes) > 1 else 0
    c = (shapes[-1]["out_ch"] * shapes[-1]["len"]) if len(shapes) > 2 else 0
    total = a + b + c

    macs = 0
    length = window
    for spec in convs:
        L = (length + 2 * spec["padding"] - spec["dilation"] * (spec["kernel"] - 1) - 1) // spec["stride"] + 1
        macs += spec["out_ch"] * L * spec["in_ch"] * spec["kernel"]
        length = L // spec["pool"] if spec["pool"] else L
    macs += shapes[-1]["out_ch"] * convs[-1]["out_ch"]  # dense layer

    return {
        "shapes": shapes,
        "buffer_a_floats": a,
        "buffer_b_floats": b,
        "buffer_c_floats": c,
        "arena_floats": total,
        "arena_bytes": total * 4,
        "macs": macs,
    }


# --------------------------------------------------------------------------- #
# C emission
# --------------------------------------------------------------------------- #
def _c_array(name: str, values, ctype: str, per_line: int = 16) -> str:
    flat = np.asarray(values).reshape(-1)
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = ", ".join(
            f"{int(v)}" if np.issubdtype(np.asarray(flat).dtype, np.integer) else f"{float(v):.9g}f"
            for v in flat[i: i + per_line]
        )
        lines.append("    " + chunk + ",")
    return f"static const {ctype} {name}[{len(flat)}] = {{\n" + "\n".join(lines) + "\n};\n"


def emit_header(info: Dict, quant: List[Dict], head_q: Dict, arena: Dict,
                norm: Dict, window: int, class_names: List[str], source: str) -> str:
    L = []
    L.append("/*")
    L.append(" * edge_model.h  —  GENERATED FILE, DO NOT EDIT BY HAND.")
    L.append(" *")
    L.append(f" * Source checkpoint : {source}")
    L.append(f" * Input window      : {window} samples x {info['n_channels']} channels")
    L.append(f" * Layers            : {len(info['convs'])} conv + 1 dense")
    L.append(f" * Weights           : int8, per-output-channel scale")
    L.append(f" * Activations       : float32 (ESP32 hardware FPU)")
    L.append(f" * Normalisation     : {norm['mode']}")
    L.append(f" * Activation arena  : {arena['arena_floats']} floats "
             f"({arena['arena_bytes'] / 1024:.1f} KB)")
    L.append(f" * MACs per window   : {arena['macs']:,}")
    L.append(" *")
    L.append(" * Regenerate with: python scripts/export_edge.py")
    L.append(" */")
    L.append("#ifndef EDGE_MODEL_H")
    L.append("#define EDGE_MODEL_H")
    L.append("")
    L.append('#include <stdint.h>')
    L.append('#include "edge_infer.h"')
    L.append("")
    L.append(f"#define EDGE_WINDOW       {window}")
    L.append(f"#define EDGE_N_CHANNELS   {info['n_channels']}")
    L.append(f"#define EDGE_N_CLASSES    {info['n_classes']}")
    L.append(f"#define EDGE_N_CONV       {len(info['convs'])}")
    L.append(f"#define EDGE_ARENA_FLOATS {arena['arena_floats']}")
    L.append(f"#define EDGE_A_FLOATS     {arena['buffer_a_floats']}")
    L.append(f"#define EDGE_B_FLOATS     {arena['buffer_b_floats']}")
    L.append(f"#define EDGE_C_FLOATS     {arena['buffer_c_floats']}")
    L.append(f"#define EDGE_INPUT_FLOATS {window * info['n_channels']}")
    L.append(f"#define EDGE_NORM_MODE    {0 if norm['mode'] == 'global' else 1}  /* 0=global, 1=per-window */")
    L.append(f"#define EDGE_HEAD_IN      {info['head_in']}")
    L.append("")

    # normalisation statistics
    if norm.get("mean") is not None:
        L.append(_c_array("EDGE_NORM_MEAN", norm["mean"], "float"))
        L.append(_c_array("EDGE_NORM_STD", norm["std"], "float"))
    else:
        L.append("/* per-window normalisation: no stored statistics needed */")
    L.append("")

    # per-conv metadata tables
    for i, (spec, shp) in enumerate(zip(info["convs"], arena["shapes"])):
        L.append(f"#define EDGE_C{i}_IN_CH    {spec['in_ch']}")
        L.append(f"#define EDGE_C{i}_OUT_CH   {spec['out_ch']}")
        L.append(f"#define EDGE_C{i}_KERNEL   {spec['kernel']}")
        L.append(f"#define EDGE_C{i}_PAD      {spec['padding']}")
        L.append(f"#define EDGE_C{i}_DIL      {spec['dilation']}")
        L.append(f"#define EDGE_C{i}_POOL     {spec['pool']}")
        L.append(f"#define EDGE_C{i}_LEN      {shp['len']}")
    L.append("")

    # weights
    for i, (spec, q) in enumerate(zip(info["convs"], quant)):
        L.append(f"static const int8_t EDGE_C{i}_W[{q['q'].numel()}] = {{")
        flat = q["q"].reshape(-1).tolist()
        for j in range(0, len(flat), 16):
            L.append("    " + ", ".join(str(int(v)) for v in flat[j: j + 16]) + ",")
        L.append("};")
        L.append(_c_array(f"EDGE_C{i}_SCALE", q["scale"].tolist(), "float"))
        L.append(_c_array(f"EDGE_C{i}_BIAS", spec["bias"].tolist(), "float"))
        L.append("")

    L.append(f"static const int8_t EDGE_HEAD_W[{head_q['q'].numel()}] = {{")
    flat = head_q["q"].reshape(-1).tolist()
    for j in range(0, len(flat), 16):
        L.append("    " + ", ".join(str(int(v)) for v in flat[j: j + 16]) + ",")
    L.append("};")
    L.append(_c_array("EDGE_HEAD_SCALE", head_q["scale"].tolist(), "float"))
    L.append(_c_array("EDGE_HEAD_BIAS", info["head_bias"].tolist(), "float"))
    L.append("")

    L.append("static const char *EDGE_CLASS_NAMES[EDGE_N_CLASSES] = {")
    for n in class_names:
        L.append(f'    "{n}",')
    L.append("};")
    L.append("")
    L.append("/* Layer table consumed by edge_infer(). Order is spatial. */")
    L.append("static const conv_spec EDGE_CONVS[EDGE_N_CONV] = {")
    for i, spec in enumerate(info["convs"]):
        L.append(f"    {{ EDGE_C{i}_IN_CH, EDGE_C{i}_OUT_CH, EDGE_C{i}_KERNEL, "
                 f"EDGE_C{i}_PAD, EDGE_C{i}_DIL, EDGE_C{i}_POOL, "
                 f"EDGE_C{i}_LEN, EDGE_C{i}_W, EDGE_C{i}_SCALE, EDGE_C{i}_BIAS }},")
    L.append("};")
    L.append("")
    L.append("#endif /* EDGE_MODEL_H */")
    return "\n".join(L) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Export a CNN checkpoint to int8 C for the ESP32")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out", type=Path, default=HEADER)
    p.add_argument("--window-samples", type=int, default=None,
                   help="override the input length (default: the checkpoint's own window)")
    p.add_argument("--report", type=Path, default=RESULTS_DIR / "edge_export.json")
    args = p.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        cands = cnn_candidates()
        if not cands:
            print("[edge] no cnn checkpoint in models/. Train one first "
                  "(scripts/train.py --models cnn --normalize window).")
            return 1
        ckpt_path = cands[-1]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("model_name") != "cnn":
        print(f"[edge] {ckpt_path.name} is a '{ckpt.get('model_name')}' model. "
              f"Only the compact CNN has a C forward pass; export a cnn checkpoint.")
        return 1

    model = CNN1DClassifier(
        int(ckpt["n_channels"]),
        int(ckpt.get("n_classes", 3)),
        **(ckpt.get("model_kwargs") or {}),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    window = args.window_samples
    if window is None:
        w = ckpt.get("window") or {}
        window = int(round(float(w.get("window_sec", 15.0)) * float(w.get("target_fs", 32.0))))
    class_names = list(ckpt.get("class_names", CLASS_NAMES))

    info = extract_cnn(model)
    quant = [quantise_int8(s["weight"]) for s in info["convs"]]
    head_q = quantise_int8(info["head_weight"])
    arena = arena_size(info["convs"], window)

    scaler = ckpt.get("scaler") or {}
    norm = {
        "mode": str(scaler.get("mode", "global")),
        "mean": scaler.get("mean"),
        "std": scaler.get("std"),
    }
    if norm["mode"] == "global" and norm["mean"] is None:
        print("[edge] WARNING: global normalisation with no stored statistics; "
              "the device cannot reproduce training input. Re-export or retrain.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(emit_header(info, quant, head_q, arena, norm,
                                    window, class_names, str(ckpt_path)))

    weights_int8 = sum(q["q"].numel() for q in quant) + head_q["q"].numel()
    weight_bytes = weights_int8 + 4 * (len(info["convs"]) * info["convs"][0]["out_ch"] + info["n_classes"])
    report = {
        "source_checkpoint": str(ckpt_path),
        "header": str(args.out),
        "window_samples": window,
        "n_channels": info["n_channels"],
        "n_classes": info["n_classes"],
        "n_convs": len(info["convs"]),
        "n_parameters": int(sum(s["weight"].numel() for s in info["convs"])
                            + info["head_weight"].numel()),
        "int8_weight_bytes": int(weights_int8),
        "weight_plus_bias_bytes": int(weight_bytes),
        "float32_weight_bytes_if_unquantised": int(4 * (
            sum(s["weight"].numel() for s in info["convs"]) + info["head_weight"].numel())),
        "arena_floats": arena["arena_floats"],
        "arena_bytes": arena["arena_bytes"],
        "arena_kb": round(arena["arena_bytes"] / 1024.0, 1),
        "macs_per_window": arena["macs"],
        "normalize_mode": norm["mode"],
        "max_quantisation_error": max([q["max_err"] for q in quant] + [head_q["max_err"]]),
        "layer_shapes": arena["shapes"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))

    print(f"[edge] {ckpt_path.name} -> {args.out}")
    print(f"       window        : {window} samples x {info['n_channels']} channels")
    print(f"       parameters    : {report['n_parameters']:,}")
    print(f"       int8 weights  : {weights_int8:,} bytes "
          f"(from {report['float32_weight_bytes_if_unquantised']:,} float32)")
    print(f"       activation arena: {arena['arena_floats']:,} floats "
          f"= {arena['arena_bytes'] / 1024:.1f} KB")
    print(f"       MACs/window   : {arena['macs']:,}")
    print(f"       max quant err : {report['max_quantisation_error']:.5f}")
    print(f"       normalisation : {norm['mode']}")
    print(f"[edge] report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
