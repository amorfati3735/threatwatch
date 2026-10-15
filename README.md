# ThreatWatch — wearable + vision distress detection

An ESP32 wearable (PPG + IMU + OLED) fused with a camera-based pose analyser,
detecting physiological stress/distress and dispatching a real alert. The model is
trained on the **real WESAD dataset** and also runs **on the board**, quantised.

This is the merge of two predecessor projects into one repository, plus the fixes
documented in [`docs/FLAWS_AND_FIXES.md`](docs/FLAWS_AND_FIXES.md).

> **Base paper:** Wagh, Sutar, Kadam, Jadhav, Yadav & Pawar, *"Evaluation of IoT
> based smart safety systems for women and children using machine learning
> techniques"*, **Scientific Reports** (2025), doi `10.1038/s41598-025-29146-4`.
> Its SVM + Naive Bayes classifiers are the baseline this project replaces. Its
> own classifiers are **run here** so the comparison is a measurement.

---

## Why this project exists

The base paper reports **99.7 %** threat-detection accuracy using SVM and Naive
Bayes. That number comes from **window-level k-fold cross-validation**, which
places windows from the *same person* in both training and test. For a wearable
that must work on someone it has never met, that is not a real accuracy.

So this project does two things at once:

1. **Replaces SVM/NB with deep sequence models** (CNN, BiLSTM, Transformer,
   CNN+Transformer, Patch-ViT) and measures all of them — including SVM and Naive
   Bayes themselves, under identical conditions.
2. **Reports two protocols**: the base paper's own window-level k-fold, for a
   like-for-like comparison, and an honest subject-disjoint split, for what the
   device actually achieves on a new wearer — plus the reason the two differ.

Headline measured result, honest protocol (subject-disjoint test, subjects S15–S17):

| model | test macro-F1 | test accuracy |
|---|---|---|
| **1-D CNN, per-window normalisation** | **0.617** | **0.709** |
| SVM (the base paper's classifier) | 0.486 | 0.512 |
| Logistic regression | 0.496 | 0.517 |
| Naive Bayes (the base paper's classifier) | 0.375 | 0.410 |
| 1-D CNN, global normalisation | 0.330 | 0.327 |

The normalisation change alone is the difference between the deep model losing to
SVM and beating it by +0.131 macro-F1. See
[`docs/TASKS_ALIGNMENT.md`](docs/TASKS_ALIGNMENT.md).

---

## Quickstart

```bash
# 1. environment
python3 -m venv /home/pratik/.venvs/threatwatch
/home/pratik/.venvs/threatwatch/bin/pip install -r requirements.txt

# 2. tests (no dataset needed, no hardware needed)
/home/pratik/.venvs/threatwatch/bin/python -m pytest tests -q      # 68 passed

# 3. the live dashboard
python run.py                       # http://localhost:8000
```

The dashboard runs with **no hardware and no trained model** — it falls back to
the WESAD replay and the heuristic, so it is always demonstrable.

### With the real dataset

WESAD is expected at `data/wesad/S*/S*.pkl` (15 subjects, S2–S17; S12 is excluded
in the official dataset).

```bash
python scripts/download_wesad.py                  # if not already present
python scripts/preprocess.py --channels both --window 15 --stride 5
python scripts/experiments.sh sweep               # pick normalisation/augmentation
python scripts/experiments.sh main                # the model comparison
python scripts/plot_results.py                    # comparison figure + table
python scripts/export_onnx.py --all               # ONNX, verified against PyTorch
```

`scripts/experiments.sh` is the reproducible driver for every number in the
report; see [`docs/ROADMAP.md`](docs/ROADMAP.md) for the phases and their measured
wall-clock costs.

---

## Hardware

ESP32 devkit + **MAX30102** (PPG) + **MPU6050** (IMU) + **SSD1306 128×64 OLED**,
on one 3.3 V I²C bus (`SDA=GPIO21`, `SCL=GPIO22`).

```bash
CLI="/mnt/c/Users/DELL/AppData/Local/Programs/Arduino IDE/resources/app/lib/backend/resources/arduino-cli.exe"
"$CLI" compile --fqbn esp32:esp32:esp32 firmware/esp32_wearable
# → Sketch uses 1123215 bytes (85%) of program storage; global vars 42884 bytes (13%)
```

Full wiring table, library list, power-up checklist and troubleshooting:
[`docs/HARDWARE_SETUP.md`](docs/HARDWARE_SETUP.md).

### The rule the firmware keeps

```
ts,hr,spo2,eda,temp,ax,ay,az,gx,gy,gz,finger,label
1.000,78.0,,,33.40,0.010,-0.020,0.990,1.00,0.30,-0.50,1,-1
```

Empty fields are **real absences**, never guesses. `spo2` and `eda` are always
blank on this kit (no GSR module; no calibrated SpO₂ curve). `hr` is blank when
there is no finger contact or too few beats. A missing reading is `None` all the
way through the pipeline and is never replaced by a plausible-looking number.

---

## Layout

```
run.py                 one-command launcher
threatwatch/           the package
  config.py            the only place constants live
  data/                SensorPacket · WESAD loader · windowing · replay · sources/
  models/              CNN · BiLSTM · Transformer · CNN+Transformer · PatchViT · baselines
  ml/                  training · calibration · metrics · inference · ONNX export
  vision/              camera · YOLOv8-pose · pose features · threaded worker
  fusion/              fusion engine · alert state machine
  alerts/              Notifier interface · Telegram · cooldown/dedup/retry dispatcher
  api/                 FastAPI pipeline runner · routes · websocket
frontend/              live dashboard
firmware/              esp32_wearable · esp32_edge_wearable · arduino_uno_wearable
scripts/               download · preprocess · train · experiments · evaluate · live
tests/                 68 tests, none of which need hardware or the dataset
docs/                  see below
```

## Documentation

| Document | What it is |
|---|---|
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | **start here** — scope, status, measured costs, what is left |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | the merged system, the data contract, both inference hosts |
| [`docs/TASKS_ALIGNMENT.md`](docs/TASKS_ALIGNMENT.md) | the professor's three tasks, answered with numbers |
| [`docs/FLAWS_AND_FIXES.md`](docs/FLAWS_AND_FIXES.md) | every defect found in the two predecessors, with evidence |
| [`docs/HARDWARE_SETUP.md`](docs/HARDWARE_SETUP.md) | wiring, libraries, flashing, troubleshooting |
| [`docs/TRAINING_PROMPT.md`](docs/TRAINING_PROMPT.md) | self-contained hand-off to finish the training runs |
| [`docs/EDGE_DEPLOYMENT.md`](docs/EDGE_DEPLOYMENT.md) | int8 C inference on the ESP32, with parity proof and measurements |
| [`docs/report-outline.md`](docs/report-outline.md) | the write-up structure against the base-paper template |

## Design rules

1. **No fabricated sensor values.** An absent reading is `None` / an empty CSV
   field, end to end.
2. **No leakage.** Subjects are partitioned before windowing; the standardizer is
   fitted on train only.
3. **No unverified claims.** Every number comes from a script in `scripts/` writing
   into `results/`, and every fix has a test or a measurement behind it.
4. **A model is not deployed because a file was copied to a board.** It is deployed
   when the on-board forward pass is shown to produce the same outputs as the
   reference implementation.
