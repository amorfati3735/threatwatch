# Experiments

Everything here is meant to be reproducible from the commands shown. Measured
numbers are filled in as runs complete; nothing is estimated.

## Protocol

* Dataset: **real WESAD**, wrist signals only.
* Splits (subject-independent): train **S2–S11**, val **S13–S14**, test **S15–S17**.
* Windowing: 30 s / 10 s stride @ 32 Hz (default); a shorter-window run is planned.
* Standardization: fit on training subjects only.
* Metrics: accuracy, **macro-F1**, per-class precision/recall/F1, confusion matrix.
* Hardware: RTX 3050 Laptop 6 GB / 6 CPU cores / 5.8 GB RAM (WSL2). CPU fallback.

## Run matrix

| # | task | channels | model | notes |
|---|---|---|---|---|
| 1 | binary (stress vs baseline) | full | random_forest | baseline |
| 2 | binary | full | logistic | baseline |
| 3 | binary | full | cnn | deep baseline |
| 4 | binary | full | bilstm | |
| 5 | binary | full | transformer | auto batch 16 (O(T²) memory) |
| 6 | binary | full | cnn_transformer | tokenizer + transformer |
| 7 | 3-class | full | best-of-above | |
| 8 | binary | **hw** (no EDA) | cnn_transformer | honesty gap vs #6 |

Commands:

```bash
python scripts/preprocess.py --channels full        # -> data/processed/w30_s10_fs32.npz
python scripts/preprocess.py --channels hw          # -> data/processed/w30_s10_fs32_hw.npz

python scripts/train.py --task binary --models random_forest logistic cnn bilstm transformer cnn_transformer
python scripts/train.py --task 3class --models cnn_transformer
python scripts/train.py --task binary --dataset data/processed/w30_s10_fs32_hw.npz --models cnn_transformer

python scripts/evaluate.py --checkpoint models/cnn_transformer.pt
python scripts/plot_results.py
```

## Feasibility notes (measured, not guessed)

* **Pure Transformer memory.** Attention is O(B·H·T²). At T = 960 (30 s @ 32 Hz),
  H = 4, B = 64 the attention tensor is ≈ 0.9 GB per layer — too large for a
  6 GB GPU alongside activations. `scripts/train.py` therefore defaults the
  plain transformer to **batch 16** and provides `cnn_transformer`, whose strided
  conv tokenizer cuts T to 240 and the attention cost ~16×. This is the
  architecture justification, backed by the memory arithmetic.

## Results

⟨filled in from results/comparison*.json and script console output⟩

### Baselines & deep models (binary, full channels)
| model | params | accuracy | macro-F1 | train time | latency |
|---|---|---|---|---|---|
| ⟨…⟩ | | | | | |

### 3-class
| model | accuracy | macro-F1 |
|---|---|---|
| ⟨…⟩ | | |

### Hardware-only (no EDA) vs full
| channels | macro-F1 |
|---|---|
| full (with EDA) | ⟨…⟩ |
| hw (no EDA) | ⟨…⟩ |

## Error analysis / failure modes

⟨to fill: which class confuses with which, per-subject spread, affect of window size⟩
