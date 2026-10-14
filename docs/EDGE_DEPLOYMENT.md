# Edge deployment — measured, not promised

How the trained CNN is prepared to run on the ESP32, what it costs, and exactly
what is and is not verified. Every number below came from a command in this
repo; the command is named next to it.

## 1. The artefact

`python scripts/export_edge.py` takes the deployed checkpoint
(`models/cnn_wnorm.pt`), folds BatchNorm into the convolutions, quantises all
weights to **int8 with per-output-channel scale**, and emits
`firmware/edge_model/edge_model.h`. Activations stay float32 because the ESP32
has a hardware single-precision FPU — this removes requantisation error while
still cutting weight storage ~4x.

| Metric | Value | Source |
|---|---|---|
| Parameters | 48,576 conv+dense (checkpoint total 49,347 incl. BatchNorm) | `results/edge_export.json` |
| int8 weight storage | 48,576 B (vs 194,304 B float32) | same |
| Activation arena | 61,952 floats = **242.0 KB** | same |
| MACs per 15 s window | 9,484,288 | same |
| Max quantisation error | 0.0033 | same |
| Normalisation | per-window z-score (reproducible on device) | same |
| Input window | 480 samples × 7 channels @ 32 Hz | checkpoint |

## 2. Correctness — `tests/test_edge_parity.py` (3 tests, passing)

The test compiles **the exact `edge_infer.c` the firmware includes**, with the
host `cc`, and compares its logits against PyTorch on 400 real held-out test
windows:

| Check | Result |
|---|---|
| C vs int8 reference implementation | max diff **0.00000** — bit-faithful to the exported model |
| C vs float PyTorch (quantisation cost) | max 0.056, mean 0.007 |
| Argmax agreement | **1.0000** — every decision identical |
| Accuracy on test windows | 0.7075 → 0.7075 (delta **+0.0000**) — int8 costs nothing |
| Host C latency | 5.8 ms/window (x86; the ESP32 will be slower — not yet measured on board) |

The suite also gates: arena fits the board budget, int8 really is >3.5x
smaller, quantisation error stays <0.02. 58/58 tests pass.

## 3. What fits on the board (measured with `arduino-cli compile`, esp32:esp32 3.3.12)

* **Sense-and-display firmware** (`firmware/esp32_wearable/`): links clean.
  1,123,215 B flash (85%), 42,884 B static globals, 284,796 B SRAM free.
* **The model, statically allocated** (probe sketch linking `edge_infer.c`):
  **does not link** — `.dram0.bss` overflowed by 172,240 B. The arena + input
  buffers (261,248 B of .bss) exceed the 124,580 B static-data region even
  though the *total* free figure looks larger; the heap-side budget in a
  minimal sketch is 305,564 B.

**Known gap (not yet done):** move the arena from static storage to
heap (`malloc` once at init) or shrink it — the numbers say heap fits, but it is
**not built or tested on a board yet**. Until that lands, claim only this:
the model's forward pass is proven correct and the C source compiles for the
ESP32 target; the full on-board inference build is outstanding.

## 4. Accuracy vs size trade-off (measured)

| Model | Params | Test macro-F1 | Arena |
|---|---|---|---|
| `cnn_wnorm` (width 64, deployed) | 49,347 | 0.6129 | 242 KB |
| `cnn_edge32` (width 32, experiment) | 13,411 | 0.5835 | 121 KB |

Halving the width halves the arena but costs 0.03 macro-F1 — keep width 64
unless the heap fix proves tight.

## 5. Reproduce

```bash
PY=/home/pratik/.venvs/threatwatch/bin/python
$PY scripts/export_edge.py --checkpoint models/cnn_wnorm.pt   # regenerate edge_model.h + results/edge_export.json
$PY -m pytest tests/test_edge_parity.py -s   # parity + arena + size gates
$PY -m pytest tests -q                    # full suite (58 tests)
```

> The export/test tools pick the **newest** `cnn*.pt` when no checkpoint is
> given. The deployed one is `cnn_wnorm.pt` — always pass `--checkpoint`
> explicitly (or re-export after training) so the header cannot silently
> become a different model.

Board compile: see `docs/HARDWARE_SETUP.md` §3 (`arduino-cli compile`).
