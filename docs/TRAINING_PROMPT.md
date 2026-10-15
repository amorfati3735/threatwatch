# Hand-off prompt: finish the model training

Copy everything below the line into the other agent/model. It is self-contained.

---

## Task

You are finishing the **model-training and evaluation phase** of an existing,
working Python project. The code, the data pipeline and the evaluation harness are
already written and verified. Your job is to **run the experiments and produce the
result artefacts** — not to redesign anything.

**Repository:** `/mnt/c/Users/DELL/threatwatch`

## Current status (as of the hand-off)

Already done — do **not** redo: per-window normalisation sweep, wnorm model
comparisons (`results/comparison_wnorm.json` — CNN best at 0.6129 macro-F1), the
edge32 ablation, and the edge-export parity harness (68/68 tests pass).

Still open (the remaining work):

* `results/comparison.json` entries for **bilstm** and **transformer** (the two
  long trainers) under `--normalize window`;
* `results/comparison_hw.json` — the hardware-only 6-channel ablation
  (`data/processed/w15_s5_fs32_hw.npz` exists); the physical BOM is
  **ESP32 + MAX30102 + MPU6050 + OLED** — no EDA/GSR, no camera — so this
  ablation *is* the deployable configuration;
* `results/comparison_binary.json` and `results/kfold5.json`;
* after any retrain of the CNN: re-run `scripts/export_edge.py` and
  `pytest tests/test_edge_parity.py` — the export test picks the newest
  `cnn*.pt`, so train the deployable CNN last or export the intended
  checkpoint explicitly with `--checkpoint`.


## Ground rules (non-negotiable)

1. **Never fabricate, hand-edit or "smooth" a metric.** Every number you report
   must come out of `scripts/train.py` writing a JSON file under `results/`. If a
   model fails to converge, that is a finding — report it.
2. **Do not weaken a test, skip a test, or add a type/lint suppression** to make
   something pass. If a test fails, report the failure and the cause.
3. **Do not change the split logic, the label mapping or the standardizer.** They
   are verified: subjects are partitioned *before* windowing (train `S2–S11`,
   val `S13–S14`, test `S15–S17`) and statistics are fitted on train only. Any
   change there invalidates the whole comparison.
4. **Do not touch** `firmware/`, `frontend/`, `threatwatch/vision/`,
   `threatwatch/fusion/`, `threatwatch/alerts/`. They are out of scope.
5. If you change a `.py` file, re-run `python -m pytest tests -q` and report the
   result.

## Environment

```bash
cd /mnt/c/Users/DELL/threatwatch
PY=/home/pratik/.venvs/threatwatch/bin/python     # already has torch 2.14.1+cpu, sklearn, onnx, onnxruntime, pytest

# sanity: the package must import and the suite must pass
$PY -m pytest tests -q
```

CPU only. No GPU. 6 cores, ~5.8 GB RAM — **do not run two training processes at
once**, they will thrash. The `/mnt/c` mount is slow for many small files.

**Long jobs must be fully detached or they get killed when your shell exits:**

```bash
setsid nohup bash scripts/experiments.sh main >/dev/null 2>&1 < /dev/null &
# then poll:
tail -f logs/experiments.log
```

Plain `nohup ... &` **does not survive**; use `setsid`.

## Data (already present, real WESAD — do not regenerate)

| Path | Contents |
|---|---|
| `data/wesad/S2 … S17/` | real WESAD pickles, 15 subjects, ~1 GB each |
| `data/processed/w15_s5_fs32.npz` | 15 s / 5 s @ 32 Hz, 7 channels: train 4354, val 880, test 1333 |
| `data/processed/w15_s5_fs32_hw.npz` | same, 6 channels (no EDA — hardware-only ablation) |
| `models/scaler.json`, `models/scaler_hw.json` | standardizers fitted on **train only** |

To regenerate (only if a file is missing — takes ~6 min):

```bash
setsid nohup $PY -u scripts/preprocess.py --channels both --window 15 --stride 5 \
  > logs/preprocess.log 2>&1 < /dev/null &
```

## The one finding you must preserve

A configuration sweep already established this (`results/sweep_summary.json`),
and everything downstream depends on it:

| normalisation | augmentation | val macro-F1 | **test macro-F1** |
|---|---|---|---|
| global (train-set stats) | off | 0.558 | 0.330 |
| global | on | 0.646 | 0.321 |
| **per-window z-score** | off | 0.589 | **0.612** |
| **per-window z-score** | on | 0.624 | **0.617** |

**Use `--normalize window`** for every remaining run. Under the honest
subject-disjoint protocol, per-window z-scoring roughly doubles test macro-F1
(0.32 → 0.62) because it cancels each subject's personal physiological offset.
The `--normalize global` numbers are kept only as the "before" column of this
comparison.

## What to run, in order

`scripts/experiments.sh` drives everything. Run the phases one at a time so a
crash does not lose the whole matrix:

```bash
# 1. model comparison, 3-class, full channels  (~35 min)
setsid nohup bash scripts/experiments.sh main >/dev/null 2>&1 < /dev/null &

# 2. Task-2 sensor ablation, hardware-only channels  (~5 min)
setsid nohup bash scripts/experiments.sh hw >/dev/null 2>&1 < /dev/null &

# 3. binary stress-vs-baseline, full channels  (~6 min)
setsid nohup bash scripts/experiments.sh binary >/dev/null 2>&1 < /dev/null &

# 4. the base paper's protocol: window-level stratified k-fold  (~20 min)
setsid nohup bash scripts/experiments.sh kfold >/dev/null 2>&1 < /dev/null &
```

Then, once phases 1–4 have finished:

```bash
$PY scripts/plot_results.py            # comparison figure + results/model_comparison.md
$PY scripts/export_onnx.py --all       # ONNX for every checkpoint, with a PyTorch-parity check
$PY scripts/benchmark_models.py        # refresh results/benchmark.json
$PY scripts/summarize_sweep.py
```

### Timing budget (measured on this machine — respect it)

| model | wall clock |
|---|---|
| 1-D CNN | ~35 s |
| Patch-ViT | ~40 s |
| CNN + Transformer | ~4.5 min |
| BiLSTM | ~5 min |
| pure Transformer | ~45 min (15-epoch budget — it attends over 480 timesteps, O(T²)) |
| all four classical baselines | ~60 s |

Total for all four phases: **under 1.5 hours**. Run them sequentially, not
concurrently.

## Artifacts you must produce

| File | What it is |
|---|---|
| `results/comparison.json` | 3-class, full channels: per-model test macro-F1 / accuracy / latency / temperature |
| `results/comparison_hw.json` | the hardware-only ablation (answers Task 2) |
| `results/comparison_binary.json` | binary stress-vs-baseline |
| `results/kfold5.json` | window-level k-fold — the base paper's own protocol |
| `results/<model>_metrics.json` | per-model: accuracy, macro-F1, per-class P/R/F1, confusion matrix, ECE before/after, best epoch, train time, inference latency |
| `results/<model>_confusion.png` | confusion matrix figure |
| `results/model_comparison.md` + `.png` | the leaderboard table and figure |
| `results/benchmark.json` | params / ms-per-batch / projected epoch time per architecture |
| `models/<model>.pt` | checkpoints, each embedding its window config, channel list and normaliser |
| `models/<model>.onnx` | verified ONNX export |

## Acceptance criteria

Before you report done, check each of these and show the evidence:

1. `python -m pytest tests -q` → **all tests pass** (expect 55+).
2. `results/comparison.json` contains an entry for every model: `svm`,
   `naive_bayes`, `random_forest`, `logistic`, `cnn`, `cnn_transformer`,
   `patch_vit`, `bilstm`, `transformer`.
3. Every checkpoint loads through the live path:
   ```bash
   $PY -c "
   from threatwatch.ml.inference import SensorMLPredictor
   p = SensorMLPredictor('models/cnn.pt')
   print(p.describe())"
   ```
   It must print the window config, channel list **and** `normalize: window`.
   If any checkpoint raises "does not record the window configuration", that
   checkpoint is stale — retrain it.
4. `python scripts/export_onnx.py --all` reports `verified: True` (max abs
   difference from PyTorch < 1e-3) for every model. A non-verified export must
   raise, not warn.
5. `results/kfold5.json` exists and its numbers are **much higher** than
   `results/comparison.json`. That contrast is the point — do not "fix" it.
6. No metric in any JSON has been edited by hand. Confirm by re-running one model
   and checking its numbers reproduce.

## Report back

Reply with a table of **test macro-F1 and accuracy** for every model across the
three protocols, plus:

* which model is best under the honest protocol, and by how much it beats SVM;
* the CNN's inference latency per 15 s window;
* any model that failed or overfitted, stated plainly;
* the exact command that produced each artefact.

## If something is broken

Do not invent a workaround that hides it. Report:

* the exact command,
* the full traceback or the failing assertion,
* what you already ruled out.

The project's standing rule is that a limitation reported is worth more than a
number guessed — the write-up depends on that being true.
