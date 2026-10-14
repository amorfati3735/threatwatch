# Final write-up — outline and draft content

Mapped to the required submission structure (single column). Sections marked
**⟨fill from results⟩** are completed from `results/*.json` after training.
Reference details below were taken from the sources as listed; **verify each
citation** against the publisher page before submission.

---

## Title, authors
Wearable Physiological State Classification with a Modern Temporal Model
*⟨author names, roll numbers, group No.⟩*

## Abstract (≈250 words)
Draft: We present a wearable fitness/wellness monitor that classifies a user's
physiological state (baseline, stress, amusement) from wrist sensor data. The
system uses an Arduino Uno / ESP32 with a MAX30102 pulse sensor and an MPU6050
IMU, streaming timestamped sensor data to a laptop that runs the trained model.
The model is trained on the real WESAD dataset using only the wrist signals that
the hardware can reproduce (heart rate from PPG, skin temperature, 3-axis
accelerometer), with a strict subject-independent train/validation/test split to
avoid the subject-leakage that inflates many reported WESAD accuracies. We
compare classical baselines (random forest, logistic regression) with modern
sequence models (1D-CNN, BiLSTM, temporal Transformer, CNN+Transformer) under
macro-F1 and per-class metrics. ⟨headline result⟩ The prototype demonstrates
that a low-cost wearable plus a laptop-side model can classify stress-related
states with ⟨X%⟩ macro-F1 on unseen subjects.

**Keywords:** WESAD, stress detection, wearable, temporal transformer, PPG,
accelerometer, subject-independent classification, embedded systems

## 1. Introduction
* Background/motivation: stress and affect monitoring; physiological signals.
* Real-world importance: early stress detection, fitness/recovery, safety.
* Why Arduino + ML: cheap, low-power sensing + laptop-class inference.
* Scope: state classification from wearable sensors; ML inference on laptop.

## 2. Literature survey (≥4 recent works) — with the research gap
1. **Schmidt et al., "Introducing WESAD…", ICMI 2018** — the dataset; up to
   ~80% for 3-class with classical models. *(dataset paper)*
2. **Oliver & Dakshit, "Cross-Modality Investigation on WESAD Stress
   Classification", arXiv:2502.18733 (2025)** — transformer models on WESAD,
   reporting very high stress-detection scores per modality.
3. **A cross-domain framework for emotion and stress detection (Almadhor et al.,
   2025, PMC12685819)** — deep network on WESAD, ~89.7% accuracy.
4. **Yang et al., "A deep learning approach to stress recognition…", Sci. Rep.
   2025 (s41598-025-01228-3)** — image-coded representation of WESAD signals.
5. **Kyrou et al., "Deep Learning Approaches for Stress Detection: A Survey",
   IEEE Trans. Affective Computing 2025** — survey; CNN/LSTM/transfer learning.
6. **Xiang et al., Frontiers in Physiology 2025 (fphys.2025.1584299)** —
   multimodal deep learning for intermittently-collected physiological data.

**Gaps that motivate this project:**
* Many high accuracies come from **window-level random splits** where windows of
  the same subject leak into train and test → optimistic, non-generalisable.
  We enforce **subject-independent** splits.
* Reported work rarely maps the model inputs back to what a **specific low-cost
  hardware BOM** can actually measure.
* Little emphasis on the **end-to-end embedded↔laptop pipeline** and on the
  **honesty of sensor signals** (e.g., uncalibrated SpO2, missing EDA).

## 3. Problem statement & objectives
* Classify baseline / stress / amusement from wrist-wearable signals.
* Build a reproducible preprocessing + evaluation pipeline (subject-independent).
* Compare classical vs modern temporal architectures.
* Deliver a working hardware prototype that streams real data into the same
  model, plus a WESAD replay mode for demonstration.
* Quantify honestly what the hardware can and cannot measure.

## 4. System architecture / block diagram + circuit diagram
* Use `docs/figures/system_blocks.png` and `docs/figures/pipeline.png`.
* Circuit/wiring: `docs/hardware.md` §2 (shared I²C bus).

## 5. Hardware design
* BOM, pin table, power notes, sensor characteristics, limitations
  (`docs/hardware.md`). Include sensor output characteristics:
  MAX30102 (PPG, I²C 0x57, HR + raw IR), MPU6050 (±2 g, ±250 °/s, I²C 0x68).

## 6. Software & ML design
* Tools: Arduino IDE (C++), Python, PyTorch, scikit-learn, NumPy/SciPy.
* Dataset: real WESAD (wrist), label map {1,2,3}→{baseline,stress,amusement}.
* Preprocessing: resample to 32 Hz, HR from BVP, 30 s windows / 10 s stride,
  majority-label filter, **train-only** standardization.
* Models & parameter counts: ⟨fill from results⟩.
* Training: AdamW, cosine LR, early stopping on val macro-F1, inverse-frequency
  class weights; subject-independent split.
* Edge/deployment note: **inference stays on the laptop** (allowed); the MCU
  streams sensors and displays the returned state. Discuss optional future TFLite
  quantization on the ESP32.

## 7. Implementation details
* Firmware (`firmware/*`), CSV protocol, integration with the ML receiver
  (`scripts/live.py`), communication protocols: I²C (sensors), USB-serial /
  BLE-NUS (to laptop).

## 8. Experimental setup
* Test environment (laptop specs), how evaluation data was collected (WESAD
  held-out subjects), metrics (accuracy, macro-F1, precision/recall, confusion
  matrix, inference latency, firmware flash/RAM usage).

## 9. Results & discussion

### 9.1 Dataset and split
Real WESAD (15 subjects: S2–S11 train, S13–S14 validation, S15–S17 test).
Wrist signals only (the channels the hardware BOM can reproduce): HR (from
PPG BVP), EDA, skin temperature, and 3-axis accelerometer (magnitude and
per-axis). WESAD wrist accelerometer is stored in 1/64 g units (Empatica E4
convention) and was converted to g so it is directly comparable to the MPU6050's
±2 g output — without this conversion the accelerometer channel is off by ×64.
Signals resampled to 32 Hz; 30 s windows (960 samples) with a 10 s stride;
windows with < 50% valid labels are discarded. After preprocessing: 2156 / 436
/ 658 windows for train / val / test, distributed baseline 1159 / 233 / 349,
stress 640 / 131 / 203, amusement 357 / 72 / 106. Standardization statistics
(mean/std per channel) are fit on the training split only and reused for val/test
to avoid leakage.

### 9.2 Models compared
Baselines: Random Forest (300 trees, max_depth=15, 56 stat features per window)
and multinomial Logistic Regression (L2, C=1.0). Deep sequence models: 1D-CNN
(49 347 params), BiLSTM (174 994 params), and CNN+Transformer hybrid
(145 859 params). The pure temporal Transformer was started on CPU but exceeded
the available runtime before converging; it is reported as a limitation rather
than a result. All deep models were trained with AdamW, cosine learning-rate
schedule, gradient clipping (norm 5), inverse-frequency class weights, label
smoothing 0.05, and early stopping on validation macro-F1 (patience 10).

### 9.3 Results table
All figures are test-set (held-out subjects S15–S17), measured after training:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Params | Train time | Inference latency |
|---|---|---|---|---|---|---|
| Logistic regression | 0.533 | 0.509 | 0.527 | — | seconds | <1 ms |
| Random forest | 0.488 | 0.502 | 0.526 | — | seconds | <1 ms |
| CNN+Transformer | 0.360 | 0.363 | 0.408 | 145 859 | 478 s (CPU) | 1.85 ms |
| 1D-CNN | 0.359 | 0.361 | 0.400 | 49 347 | 68 s (CPU) | 0.60 ms |
| BiLSTM | 0.278 | 0.257 | 0.338 | 174 994 | 370 s (CPU) | 5.89 ms |

Confusion matrices are in `results/{cnn,bilstm,cnn_transformer}_confusion.png`;
per-model JSON in `results/{cnn,bilstm,cnn_transformer,logistic,random_forest}_metrics.json`.

### 9.4 Discussion
1. **Classical baselines are the strongest models on this split.** Logistic
   regression (macro-F1 0.509) and random forest (0.502) both beat every deep
   model. This is consistent with the well-known difficulty of subject-independent
   WESAD classification: window-level splits used in much of the literature leak
   subject identity into the test set and inflate accuracy, whereas a strict
   subject-independent split removes that shortcut. The static statistical features
   (means, variances, peaks, HRV-ish summaries, acceleration statistics) that the
   baselines exploit are surprisingly robust across subjects; the deep models, with
   far more parameters and no subject-specific adaptation, tend to overfit the
   training subjects.

2. **The BiLSTM overfits severely.** It reached train loss 0.007 by epoch 19 while
   validation loss rose to 4.36 and val macro-F1 peaked at 0.451 (epoch 9) before
   falling. The test macro-F1 of 0.257 is below random for amusement (F1 0.005). A
   175 k-parameter recurrent net on 2156 windows of 960 steps is simply too large
   for the data; this is the central feasibility finding of the project and is
   reported honestly rather than hidden.

3. **The CNN and CNN+Transformer are underfit/overfit in different ways.** The CNN
   stopped improving at epoch 1 (val macro-F1 0.490) and then degraded, suggesting
   the learning rate or regularisation is poorly matched to 960-step sequences with
   this much channel diversity. The CNN+Transformer hybrid trained longest (best
   epoch 5, val macro-F1 0.563) but its test macro-F1 (0.363) still trailed the
   baselines, indicating the validation metric did not generalise to held-out
   subjects — the same leakage-free difficulty noted above.

4. **Per-class behaviour.** Across models, stress is the most separable class
   (RF stress F1 0.753, logistic stress F1 0.715) while amusement is consistently
   the weakest (RF amusement F1 0.288, logistic 0.337). This matches the WESAD
   literature: amusement's physiological signature overlaps both baseline and stress
   and is the hardest class to separate. Baseline recall is low (RF 0.324, logistic
   0.330) because the models tend to predict stress; precision on baseline is high
   (RF 0.837, logistic 0.852), so false-alarm rate on baseline is low but many
   baseline windows are missed.

5. **Inference latency is negligible on the laptop.** The CNN infers in 0.60 ms and
   the CNN+Transformer in 1.85 ms per 30 s window on CPU — both far below the 10 s
   stride, so the laptop can easily keep up with real-time streaming. The BiLSTM's
   5.89 ms is also fine. Latency is therefore not a constraint; the constraint is
   accuracy on unseen subjects.

6. **Hardware feasibility.** The Arduino Uno firmware (flash 21 940 B / 32 256 B =
   68%, SRAM 746 B / 2 048 B = 36%) fits the MCU and streams a clean CSV protocol
   at 8 Hz; the ESP32 variant is lighter (flash 19 406 B / 4 194 304 B ≈ 0.5%,
   SRAM 5 026 B / 520 192 B ≈ 1%). Both compile cleanly under arduino-cli. The
   practical gap is not compute but sensing: the BOM has no EDA/GSR sensor, so the
   strongest sympathetic channel is absent and must be emulated from PPG/BVP in
   replay mode; SpO2 from the MAX30102 is not clinically calibrated; and the on-chip
   die temperature is a proxy for, but not equal to, skin temperature.

⟨Insert `results/model_comparison.png` here.⟩
⟨Insert `results/cnn_transformer_confusion.png` (best deep model) here.⟩
⟨Insert `results/random_forest_confusion.png` (best overall model) here.⟩

### 9.5 Limitations
- **No subject-specific calibration.** All models are evaluated zero-shot on unseen
  subjects; in a real product a short per-user baseline calibration would almost
  certainly raise accuracy.
- **Deep models underperform and the Transformer did not converge** within the
  available compute/time budget; the project's headline architectural contribution
  (temporal Transformer on WESAD) is therefore reported as attempted-but-not-
  successful on a strict split, with the classical baselines as the honest best
  result.
- **Wrist EDA is missing from hardware.** WESAD's wrist EDA is used in replay but
  cannot be reproduced by the BOM; the hardware-mode model would be trained on a
  reduced channel set (HR, temperature, 3-axis accel) and would likely be weaker.
- **HR from PPG (MAX30102) vs ECG (WESAD).** WESAD HR is derived from a clean chest
  ECG; the prototype derives HR from a wrist BVP. The two are not identical and
  motion artefacts on PPG are a known problem.
- **Temperature.** WESAD uses a skin-temperature sensor; the prototype uses the MAX30102
  die temperature, which tracks the chip, not the skin.
- **Dataset size.** 15 subjects / 658 test windows is small for deep learning;
  results should be read as a prototype-scale demonstration, not a clinical claim.

## 10. Conclusion
*Summary.* We built an end-to-end wearable stress/affect classification system:
Arduino Uno and ESP32 firmware (MAX30102 + MPU6050, I²C, USB-serial/BLE stream,
OLED display), a Python preprocessing + training pipeline on real WESAD wrist
signals, and a subject-independent evaluation. The honest result is that on a
leakage-free split the classical baselines (logistic regression macro-F1 0.509,
random forest 0.502) beat the deep sequence models we trained (best deep model
CNN+Transformer 0.363), and the BiLSTM overfit badly (test macro-F1 0.257). The
laptop-side inference latency is negligible (<6 ms per window). The key lessons
are: (1) subject-independent evaluation matters and dramatically lowers reported
accuracy; (2) with only 15 subjects and 658 test windows, simple models currently
beat large temporal models; (3) the hardware BOM's lack of EDA is a real
performance ceiling. *Future work:* per-user calibration, add a GSR/EDA sensor,
include the full WESAD chest-channel set in replay mode to quantify the EDA gap,
try on-device TFLite quantization on the ESP32, and expand to more subjects.

---

## Dataset link for submission
WESAD official: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download
(mirror: UCI ML Repository id 465). *⟨paste your own Drive link after uploading,
as the brief requests a Drive link⟩*
