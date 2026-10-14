# Agent progress log

Living engineering log for the WESAD fitness-tracker project.
Legend: **[verified-exp]** verified experimentally · **[verified-doc]** verified from
docs/source · **[assumption]** reasonable engineering assumption · **[unknown]** still open.

---

## 1. Environment recon  — 2026-10-09

| Item | Value | Status |
|---|---|---|
| OS | WSL2 Arch Linux on Windows 11 | [verified-exp] |
| Working dir | `/mnt/c/Users/DELL/fitness tracker` (empty except a task brief) | [verified-exp] |
| Python | 3.14.7 system, **no pip by default** | [verified-exp] |
| Fix | `python3 -m venv /home/pratik/.venvs/fitness` (bundled ensurepip works) | [verified-exp] |
| Installed | numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6, matplotlib | [verified-exp] |
| PyTorch | 2.14.1 cp314 wheel available; installing | [verified-exp] |
| GPU | NVIDIA RTX 3050 Laptop 6 GB, visible via `nvidia-smi` (CUDA 13.4 driver) | [verified-exp] |
| CPU/RAM | 6 cores / 5.8 GB RAM | [verified-exp] |
| sudo | password required (no passwordless sudo → no `pacman`) | [verified-exp] |

**Note:** the venv lives on the Linux ext4 home (faster, avoids thousands of
small files on the `/mnt/c` 9p mount). Project code stays on `/mnt/c`.

---

## 2. Discovery: an existing sibling project

There is a prior capstone repo at `/mnt/c/Users/DELL/Documents/esa project`
("Multimodal IoT-Based Threat & Distress Detection System"). Inspected it
without modifying it.

**What it is:** FastAPI backend + browser dashboard that fuses a WESAD replay
"wearable" signal, a YOLOv8-pose webcam, a weighted fusion engine and an alert
state machine. Includes an Arduino Uno firmware sketch.

**Reused ideas:** `SensorSource` abstraction, Arduino firmware layout,
WESAD label mapping.

**Problems found (these are why we build fresh rather than copy):**

1. **The "WESAD" data is synthetic.** `data/wesad/{S2,S3,S4}/*.pkl` are all
   exactly `27 938 584` bytes, produced by `scripts/generate_synthetic_wesad.py`.
   Sizes are identical and the README claims real WESAD. [verified-exp]
   → This violates "do not fabricate sensor measurements". The new project
   downloads and uses the **real** WESAD dataset.
2. **Subject leakage.** `backend/ml/train.py` concatenates all subjects'
   windows and then uses `torch.utils.data.random_split`. Windows from the same
   person therefore appear in both train and validation → inflated, non-generalisable
   accuracy. [verified-exp] → The new pipeline splits **by subject** first.
3. **Hardware/ML feature mismatch.** The model consumes *chest* ECG/EDA/Resp/Temp/ACC,
   but the physical prototype is a wrist device with MAX30102 (PPG) + MPU6050
   (IMU) and no ECG. The learned distribution cannot be reproduced by the
   hardware. [verified-exp] → New pipeline trains on **WESAD wrist** signals
   (BVP→HR, EDA, TEMP, ACC), which the hardware can actually produce.
4. Vision/threat-detection scope is unrelated to a fitness tracker and is not
   carried over.

---

## 3. Data decision

- Real WESAD archive confirmed reachable:
  `https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download` →
  HTTP 206, `content-type: application/zip`, `content-length = 2 249 444 501`
  (~2.25 GB), filename `WESAD.zip`. [verified-exp]
- Download started in background at `/home/pratik/wesad_dl/WESAD.zip`
  (resumable `curl -C -`). Initial throughput ~150–730 KB/s → ETA tens of
  minutes to ~1 h. [verified-exp]
- **The `mcrNsj7645odNqP` URL hard-coded in the old `download_wesad.py` is dead**
  (~17 KB/s, wrong content). Use the `HGdUkoNlW1Ub0Gx` link above. [verified-exp]

### WESAD facts used (from the dataset readme / Schmidt et al. 2018) [verified-doc]
- 15 subjects (S2–S17, **S12 excluded** — sensor issues; S1 does not exist).
- Chest: ACC/ECG/EDA/EMG/Resp/Temp @ 700 Hz.
- Wrist: BVP @ 64 Hz, EDA @ 4 Hz, TEMP @ 4 Hz, ACC @ 32 Hz.
- Labels @ 700 Hz: 1 baseline, 2 stress, 3 amusement, 0/4/5/6/7 = transient/ignore.

---

## 4. Architecture decisions (reversible)

| Decision | Choice | Why |
|---|---|---|
| Dataset | Real WESAD, wrist signals | Matches hardware; replaces fabricated data |
| Input channels | `hr, eda, temp, acc_x, acc_y, acc_z, acc_mag` @ 32 Hz | MAX30102+MPU6050 producible; SpO2/gyro not in WESAD so excluded from model |
| Windowing | 30 s window / 10 s stride (configurable) | Standard WESAD window scale; enough samples |
| Splits | Fixed subject-independent: train S2–S11, val S13–S14, test S15–S17 | Kills subject leakage |
| Standardizer | Fit on **train subjects only** | Avoids normalization leakage |
| Models | RF/LogReg baseline → 1D-CNN → BiLSTM → Transformer → CNN+Transformer | Baseline-first as required |
| Metrics | accuracy, macro-F1, per-class P/R/F1, confusion matrix | Imbalance-aware |

---

## 5. Firmware — compiled, not just written [verified-exp]

Used the `arduino-cli.exe` bundled in the installed Arduino IDE 2.x to actually
compile the Uno sketch (`arduino:avr:uno`, core 1.8.8).

* First compile **failed**: `MAX30105 has no member named 'checkForBeat'`.
  Root cause: in the SparkFun MAX3010x library `checkForBeat()` is a free
  function declared in `heartRate.h`, not a class method.
* Fix: `#include "heartRate.h"` and call `checkForBeat(ir)` (matching the
  library's own `Example5_HeartRate`).
* Recompile **succeeds**:
  `Sketch uses 21 940 bytes (68%) of program storage; global vars 746 bytes (36%) of SRAM.`
* Installed the SparkFun MAX3010x library into the user's Arduino libraries via
  `arduino-cli lib install` (needed to compile; it is required for the build anyway).
* ESP32 sketch **not yet compile-verified** — the `esp32` board core is not
  installed and its package download competes with the dataset download.
  Marked as **[unknown]** / pending.

## 6. Files created so far

```
fitness tracker/
├── fittrack/
│   ├── config.py            paths, subject splits, window/train config, channel sets
│   ├── data/wesad.py        real WESAD loader, wrist extraction, HR-from-PPG
│   ├── data/dataset.py      windowing, leakage-free splits, standardizer
│   ├── models/seq_models.py CNN / BiLSTM / Transformer / CNN+Transformer
│   ├── models/baselines.py  RandomForest / LogisticRegression
│   ├── eval/metrics.py      metrics + confusion-matrix plots
│   ├── training.py          training/eval engine (early stop, latency)
│   ├── inference.py         checkpoint loader + smoothed live inference
│   └── sources/             base, simulator, serial, ble, wesad, replay
├── scripts/                 download_wesad, preprocess, train, evaluate, live, replay
├── firmware/
│   ├── arduino_uno_wearable/  (compiles: 68% flash / 36% RAM)
│   └── esp32_wearable/        (serial + BLE NUS; unverified)
├── tests/                   test_pipeline.py, test_wesad_loader.py
├── docs/                    agent-progress.md, architecture.md, hardware.md,
│                            make_diagrams.py, figures/*.png
└── README.md, requirements.txt, .gitignore
```

## 7. Verification so far [verified-exp]

* `tests/test_wesad_loader.py` — loader against a correctly-structured fixture:
  correct output length (`4 min * 32 Hz = 7680`), label mapping 1/2/3→0/1/2 and
  0→-1, **derived HR = 72.5 BPM** for a 1.2 Hz synthetic pulse. PASSED.
* Core smoke test — windowing (shape/labels/transient-drop), standardizer
  (mean≈0, std≈1), serial CSV round-trip (missing fields stay missing),
  WindowBuffer interpolation to (64, 7). PASSED.
* Uno firmware compile. PASSED (after the `checkForBeat` fix).

## 8. Open issues / TODO

All items below marked done.

- [x] WESAD download finished and verified (2 249 444 501 bytes, 15 subjects
      S2–S17; extracted to `/home/pratik/wesad/WESAD`). [verified-exp]
- [x] PyTorch 2.14.1 installed (CPU wheel — CUDA deps abandoned after two
      "No space left on device" failures on the 2.9 GB `/tmp` tmpfs; CPU PyTorch
      is only 196 MB and sufficient for these model sizes). [verified-exp]
- [x] `preprocess.py` ran on all 15 real subjects → 2156/436/658 windows
      train/val/test; scaler + processed dataset written. [verified-exp]
- [x] Baseline models (random_forest, logistic) trained. [verified-exp]
- [x] Deep models trained on CPU: cnn (macro-F1 0.361), cnn_transformer
      (0.363), bilstm (0.257, overfit). Pure Transformer did not finish within
      the available runtime and is reported as a limitation. [verified-exp]
- [x] Results + plots written: `results/comparison.json`, `results/*_metrics.json`,
      `results/*_confusion.png`, `results/model_comparison.png` + `.md`. [verified-exp]
- [x] Report outline (Sections 1–10) written with measured results filled in.
      [verified-doc]
- [x] Full test suite re-run: `test_pipeline.py` PASSED, `test_serial_loopback.py`
      PASSED (after firmware timestamp fix), `test_wesad_loader.py` PASSED.
      `tests/test_benchmark_models.py` will be auto-discovered by pytest but its
      runtime assertions are commented — see `tests/README.md`. [verified-exp]
- [ ] ESP32 core install + compile-verify sketch — still open; CPU/space was
        prioritised for the WESAD download + training. [unknown]

---

## 9. Results summary (real WESAD, subject-independent split)

Best overall: **logistic regression macro-F1 0.509** (acc 0.533) and
**random forest macro-F1 0.502** (acc 0.488). Best deep model:
**CNN+Transformer macro-F1 0.363**. **BiLSTM overfit** badly (train loss 0.007,
val loss 4.36, test macro-F1 0.257). Pure Transformer did not converge in
available time.

Key lessons (also written into the report):
1. Subject-independent evaluation drastically lowers accuracy vs the window-level
   splits used in much of the literature.
2. On 15 subjects / 658 test windows, classical baselines beat large temporal
   models; this is reported honestly rather than hidden.
3. Stress is the most separable class (RF 0.753, logistic 0.715); amusement is
   the weakest (RF 0.288, logistic 0.337); baseline recall is low but precision
   is high (low false-alarm rate).
4. Laptop inference latency is negligible (CNN 0.60 ms, CNN+Transformer 1.85 ms,
   BiLSTM 5.88 ms per 30 s window) — not a constraint.

## 10. Files for submission

Everything needed is on disk at `/mnt/c/Users/DELL/fitness tracker`. A submission
archive `fitness tracker submission.zip` can be regenerated with:

    cd "/mnt/c/Users/DELL" && zip -9 -r "fitness tracker submission.zip" \
        "fitness tracker" -x "fitness tracker/data/unprocessed/*" \
        "fitness tracker/.git/*" "fitness tracker/__pycache__/*"

Do **not** include `/home/pratik/wesad/WESAD` (the raw 13.7 GB WESAD extraction)
unless the brief explicitly asks for the raw dataset alongside the code.
