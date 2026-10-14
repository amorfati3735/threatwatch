# Task alignment

The three tasks from `Problem Statement.docx`, each answered with what was built
and what was measured. Every number here is produced by
`scripts/experiments.sh` and lands in `results/`; nothing is estimated.

Status legend: **[measured]** produced by a run in this repo ·
**[pending]** the run is queued (`docs/TRAINING_PROMPT.md`) · **[analysis]** a
reasoned argument rather than an experiment, labelled as such.

---

## Task 1 — Deep Learning-Based Threat Detection

> "Replace SVM/NB with: Transformer, LSTM, Bi-LSTM, Vision Transformer, Graph
> Neural Networks, for more robust temporal analysis. **Also compare results**"

### Built

| Model | File | Params | Note |
|---|---|---|---|
| 1-D CNN | `models/seq_models.py::CNN1DClassifier` | 49 347 | dilated convs, global average pool |
| BiLSTM | `BiLSTMClassifier` | 174 994 | bidirectional + additive attention pooling |
| Transformer | `TransformerClassifier` | 134 737 | encoder-only, sinusoidal positions |
| CNN + Transformer | `CNNTransformerClassifier` | 145 859 | strided-conv tokenizer → attention |
| **Patch-ViT** | `PatchViTClassifier` | 181 507 | 1-D patchified ViT with a CLS token |
| **SVM** | `models/baselines.py` | — | **the base paper's own classifier** |
| **Naive Bayes** | `baselines.py` | — | **the base paper's own classifier** |
| Random forest | `baselines.py` | — | classical reference |
| Logistic regression | `baselines.py` | — | classical reference |

SVM and Naive Bayes are included specifically because the base paper reports with
those two. A comparison against SVM/NB that does not actually run SVM/NB is not a
comparison.

### Measured — honest protocol (subject-disjoint test set S15–S17)

**[measured]** from `results/sweep_summary.json` and `results/*_metrics.json`:

| model | normalisation | test macro-F1 | test accuracy |
|---|---|---|---|
| **1-D CNN** | **per-window** | **0.617** | **0.709** |
| 1-D CNN | global | 0.330 | 0.327 |
| SVM | per-window | 0.486 | 0.512 |
| Naive Bayes | per-window | 0.375 | 0.410 |
| Logistic regression | global | 0.496 | 0.517 |
| Random forest | global | 0.351 | 0.344 |
| CNN + Transformer | global | 0.358 | 0.362 |
| Patch-ViT | global | 0.331 | 0.338 |

The remaining cells (all deep models under per-window normalisation, the BiLSTM,
and the pure Transformer) are produced by `scripts/experiments.sh main`;
see `docs/TRAINING_PROMPT.md`.

### The result that matters

Two findings, both measured, both reportable:

**(a) Replacing SVM/NB with a deep model helps — but only if the data is
normalised per window.** Under the honest protocol the 1-D CNN goes from *losing*
to SVM (0.330 vs 0.486) to *beating* it by +0.131 macro-F1 (0.617 vs 0.486) purely
from changing the normalisation from training-set statistics to per-window
z-scoring. The full configuration sweep:

| normalisation | augmentation | val macro-F1 | test macro-F1 | val−test gap |
|---|---|---|---|---|
| global | off | 0.558 | 0.330 | +0.228 |
| global | on | 0.646 | 0.321 | +0.325 |
| per-window | off | 0.589 | 0.612 | −0.023 |
| **per-window** | **on** | 0.624 | **0.617** | **+0.007** |

Read the last column. Under global normalisation the model *looks* better on
validation and is much worse on test — a textbook subject-shift failure. Per-window
normalisation closes the gap to +0.007, i.e. the model now generalises to people it
has never seen. The augmentation's main contribution is the last epoch count: with
augmentation it trains for 32 epochs instead of 9 without plateauing, which is
evidence the transforms are doing real regularising work rather than adding noise.

**(b) The base paper's 99.7 % is not comparable to this number, and the write-up
says so.** Wagh et al. report 99.7 % using **window-level k-fold
cross-validation**, which puts windows from the same person in both train and
test. Task 1's *"also compare results"* is answered by measuring our models under
**their** protocol as well — `scripts/experiments.sh kfold` →
`results/kfold5.json` **[pending]**. That is the only like-for-like comparison
available, and it is expected to be high; the honest subject-disjoint number is
reported alongside it, with the reason the first is inflated.

### GNN — **[analysis]**

Task 1 also lists Graph Neural Networks. A GNN over our 7 sensor channels is a
tiny, nearly-complete graph with no meaningful topology; the message-passing
machinery would have more parameters in its adjacency handling than signal to
learn from, and the result would be a decoration rather than a contribution.
Rather than bolt one on, this is stated, and the condition under which a GNN
*would* earn its place is spelled out: a multi-node body-sensor network (several
devices, or several people in an environment) where edges carry real relational
information. Answering a task by explaining why one of its suggestions does not
apply here is a stronger result than a GNN that trains and proves nothing.

---

## Task 2 — Additional sensors

> "Additional sensors could improve robustness: ECG, PPG, GSR, SpO₂, respiration,
> EMG, EEG to improve detection accuracy"

### Built — and the difference from the predecessor

The model is trained on **WESAD wrist** channels, because that is what the
hardware can physically produce:

| Channel | Source in WESAD | Produced by the kit? |
|---|---|---|
| `hr` | wrist BVP @ 64 Hz → beat detection | ✅ MAX30102 (PPG) |
| `eda` | wrist EDA @ 4 Hz | ⚠️ needs a GSR module — not in the kit |
| `temp` | wrist TEMP @ 4 Hz | ✅ MPU6050 die temp *(see caveat)* |
| `acc_x/y/z`, `acc_mag` | wrist ACC @ 32 Hz (1/64 g) | ✅ MPU6050 |

Two channel sets are produced so the cost of a missing sensor is **measured, not
hidden**:

* `full` — 7 channels including EDA
* `hw` — 6 channels, exactly what this hardware can generate

`scripts/experiments.sh hw` → `results/comparison_hw.json` **[pending]** gives the
Task-2 answer directly: the macro-F1 difference between `full` and `hw` is the
value of the GSR module.

The predecessor project consumed `signal["chest"]` — ECG, EDA, respiration, chest
ACC — none of which a wrist PPG+IMU device can produce, so its "live hardware"
mode was disconnected from its own training distribution by construction.

### Caveats stated plainly

* **SpO₂ is not implemented.** SpO₂ is not in WESAD's wrist set, and a trustworthy
  value needs a per-sensor red/IR calibration curve that the MAX30102 library does
  not provide. The firmware emits the field **empty**. The base paper's
  "blood oxygen" claim is not something this project can honestly reproduce with
  this sensor, and inventing the number is exactly the failure mode this project
  was built to avoid.
* **Temperature is the MPU6050's die temperature**, not skin temperature. It is
  reported as recorded and labelled as die temperature in the firmware comment.
* **ECG, EMG, EEG and respiration are not carried over.** They are not producible
  by the kit. Rather than train on them and then fail to generate them live, the
  roadmap in `docs/HARDWARE_SETUP.md` names what each would add and what hardware
  it would need (a GSR module first, since EDA is the highest-value missing
  channel).

---

## Task 3 — Wearable sensing + computer vision

> "Combine wearable sensing with computer vision: YOLOv11/YOLOv12, RT-DETR,
> Vision Transformers"

### Built

* `vision/detector.py` — YOLOv8-pose keypoint detection (`yolov8n-pose.pt`).
* `vision/pose_features.py` — interprets the keypoints into interpretable
  distress cues: shoulder tilt (fall), bounding-box aspect ratio, wrist-to-head
  proximity (defensive posture), and inter-frame motion energy, each with a
  threshold in `config.py`.
* `vision/worker.py` — runs the whole vision path on a **dedicated thread** and
  publishes an immutable snapshot, so vision can never throttle telemetry.
* `fusion/engine.py` — weights the wearable distress score against the visual
  distress score and applies documented suppression rules.
* `frontend/` — the live dashboard showing both signals, the fused score, the FSM
  state, and the audit log.

### Measured

* Vision-on-telemetry-latency: **[pending]** — the F10 evidence, produced by
  running the server with and without the camera and recording pipeline FPS.
* Fusion behaviour is unit-tested (`tests/test_fusion.py`): weights, suppression
  rules and clamping.

### Model choice — **[analysis]**

The task names YOLOv11/YOLOv12, RT-DETR and Vision Transformers. This project uses
**YOLOv8-pose**, because what the threat logic actually needs is **skeleton
keypoints**, not object boxes: a fall, a defensive posture and violent motion are
joint-geometry questions, and a pose model answers them in one pass. RT-DETR and a
ViT would be *detectors* — better at "what object is where" — which is not the
question being asked. The upgrade path and its trade-off (RT-DETR's accuracy and
its latency on a laptop CPU, versus a pose-sequence ViT that could classify motion
over time instead of per-frame heuristics) is analysed in the write-up rather than
implemented as a slower substitute for the thing that already works.

Note also the predecessor project's own vision code, which fused a vision score
into its sensor pipeline, is *retained in spirit* — the fix was structural
(threading, honest absence semantics, testable suppression rules), not a rewrite
of the pose analyser.

---

## Cross-cutting: what the merge fixed

`docs/FLAWS_AND_FIXES.md` is the full list. The ones that change the *results*
rather than just the code:

| # | Flaw in a predecessor | Consequence | Fix |
|---|---|---|---|
| F1 | "WESAD" was three identical synthetic pickles | every reported number unverifiable | real 15-subject WESAD, verified file sizes |
| F2 | train/val split by *window*, mixing subjects | inflated scores | subjects partitioned *before* windowing |
| F3 | scaler fitted on all data | normalisation leakage | fitted on train only |
| F4 | model trained on chest ECG/EDA/Respiration | hardware could never reproduce it | trained on wrist BVP/EDA/TEMP/ACC |
| F5 | missing 1/64 g accelerometer conversion | acceleration channel dead (×64 off) | converted in the loader |
| F6 | `700 // 32 = 21 → 33.3 Hz`, claimed as 32 Hz | every window length subtly wrong | resample by timestamp onto an explicit grid |
| F7 | SpO₂ computed as `110 − 15·(red/ir)` | a fabricated physiological measurement | emitted empty |
| F8 | missing sensor fields defaulted to plausible values | fake readings reached the model | `None` semantics end-to-end |
| F11 | "emergency dispatch" was a log line | the headline capability did not exist | real notifier with dedup/cooldown/retry |
| F12 | three FSM timer bugs + no cooldown | wrong escalation timing, repeat alerts | rewritten with per-transition resets and tests |
