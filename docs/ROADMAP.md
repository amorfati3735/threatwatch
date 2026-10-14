# ROADMAP — what this project is, where it stands, and what is left

Last updated: 2026-10-12.  Read this first; it is the master checklist.

---

## 1. What the project actually is (from the two source documents)

**`Problem Statement.docx`** (the professor's three tasks):

| Task | Verbatim | What it means |
|---|---|---|
| 1 | "Deep Learning-Based Threat Detection. Replace SVM/NB with: Transformer, LSTM, Bi-LSTM, Vision Transformer, Graph Neural Networks, for more robust temporal analysis. **Also compare results**" | Reproduce the reference paper's task, but swap its SVM/Naive-Bayes for deep sequence models and report a measured comparison. |
| 2 | "Additional sensors could improve robustness: ECG, PPG, GSR, SpO₂, respiration, EMG, EEG to improve detection accuracy" | Choose which extra sensors are worth adding and *quantify* what each one buys. |
| 3 | "Combine wearable sensing with computer vision: YOLOv11/YOLOv12, RT-DETR, Vision Transformers" | Fuse the wearable physiological signal with a camera-based visual threat detector. |

**`reference paper Children Safety(1).pdf`** = Wagh, Sutar, Kadam, Jadhav, Yadav & Pawar,
*"Evaluation of IoT based smart safety systems for women and children using machine
learning techniques"*, **Scientific Reports** (Nature), doi `10.1038/s41598-025-29146-4`, 31 pp.

That is the **base paper** the write-up must be templated on. Its headline numbers,
which our project exists to beat / examine:

| Base paper claim | Its setup |
|---|---|
| SVM **99.7 %** average accuracy in threat detection | k-fold cross-validation (window-level, not subject-independent) |
| Detection latency **3 s** | — |
| Battery **18–20 h** | Arduino/Raspberry Pi + GPS + GSM |
| Classifiers: **SVM + Naive Bayes** | exactly the two the professor wants replaced |
| Sensors: heart rate, temperature, GPS, accelerometer | no camera, no PPG/GSR |

**The user's reading is correct.** Three components:

1. **Software / ML** — build a model that beats the base paper's SVM/NB (Task 1) and
   measures the extra-sensor payoff (Task 2).
2. **Hardware** — an ESP32 with sensors worn on the arm/forearm, standing in front of
   the camera (the professor's verbal addition).
3. **Deployment** — put the better model *on that hardware* and demonstrate threat
   detection live during the presentation (Task 3 lives here too: wearable + vision).

Plus the **write-up** (6 marks) in the base-paper template, **non-two-column**,
sections 1–11, **due 15/10/2026 (even groups) or 16/10/2026 (odd groups)**, and a
**drive link to the dataset**.

---

## 2. Where we stand right now

Legend: ✅ done & verified · 🟡 in progress · ⬜ not started · ❗ gap found today

### 2.1 Merged codebase

| # | Item | Status |
|---|---|---|
| 1 | Two predecessor projects merged into **one** repo at `threatwatch/` (old `fittrack/` + `backend/` packages gone) | ✅ |
| 2 | One `threatwatch/config.py` — single source of truth for paths, splits, thresholds, channels | ✅ |
| 3 | One sensor contract (`SensorPacket`, `None` = not measured) shared by WESAD / serial / BLE / simulator / log-replay | ✅ |
| 4 | Dead code removed: `scripts/replay.py` (superseded by `live.py --source replay`), unreferenced legacy `.npz` | ✅ |
| 5 | `utils/logging.py` import side-effect removed; audit trail now persisted to `logs/events.jsonl` | ✅ |
| 6 | **55 tests pass** (loader, sizing, serial loopback over a pty, replay engine, fusion, FSM, alerts, pipeline) | ✅ |
| 7 | `scripts/benchmark_models.py` ported & run — real feasibility numbers in `results/benchmark.json` | ✅ |
| 8 | `README.md` still describes the *old* `fittrack/` project | ❗⬜ |

### 2.2 Data

| # | Item | Status |
|---|---|---|
| 9 | **Real** WESAD (15 subjects S2–S17), not the predecessor's fabricated data | ✅ |
| 10 | Leakage-free splits: train S2–S11 / val S13–S14 / test S15–S17, proven disjoint | ✅ |
| 11 | Standardizer fitted on **train only**; persisted per channel-set | ✅ |
| 12 | Windowing 15 s / 5 s @ 32 Hz, both channel sets: full `(4354/880/1333, 480, 7)` and hardware-only `(…, 480, 6)` | ✅ |
| 13 | No NaNs, no held-out subject leakage, verified by script | ✅ |

### 2.3 Models

| # | Item | Status |
|---|---|---|
| 14 | Architectures implemented: 1D-CNN, BiLSTM, Transformer, CNN+Transformer, **Patch-ViT** (1-D ViT), RF/LogReg baselines | ✅ |
| 15 | Feasibility measured, not guessed (`results/benchmark.json`) | ✅ |
| 16 | Warmup + cosine LR, grad clipping, label smoothing, class weights, early stopping, temperature calibration + ECE | ✅ |
| 17 | 3-class, full channels, all 7 models — **training now** | 🟡 |
| 18 | 3-class, hardware-only channels (the Task-2 sensor ablation) | ⬜ |
| 19 | Binary stress-vs-baseline (the literature-standard task) | ⬜ |
| 20 | Honest discussion of GNN applicability (17-channel sensor graph is too small/weakly structured to justify bolting on) | ⬜ |

### 2.4 Live system

| # | Item | Status |
|---|---|---|
| 21 | FastAPI server + WebSocket telemetry + `python run.py` one-command launch | ✅ |
| 22 | Fusion engine with documented, unit-tested suppression rules | ✅ |
| 23 | Alert FSM rewritten (timer resets, cooldown, no phantom escalation) | ✅ |
| 24 | **Real alert dispatch**: pluggable `Notifier`, Telegram backend, cooldown + dedup + retry + audit | ✅ |
| 25 | Vision runs on its own thread (never blocks telemetry) | ✅ |
| 26 | Cyber dashboard: live charts, log panel, replay controls, demo scenarios | ✅ |
| 27 | End-to-end run through the dashboard with the trained model | ⬜ |

### 2.5 Hardware

| # | Item | Status |
|---|---|---|
| 28 | Arduino **Uno** sketch: `arduino-cli compile` **passes** (71 % flash, 39 % RAM) | ✅ |
| 29 | ESP32 sketch (serial + BLE NUS, same CSV protocol) written | ✅ |
| 30 | ESP32 core installed; sketch `arduino-cli compile`-verified | 🟡 (downloading) |
| 31 | SpO₂ / EDA emitted **empty** rather than fabricated | ✅ |

### 2.6 ❗ The big gap — the model is not actually ON the hardware

| # | Item | Status |
|---|---|---|
| 32 | Today the ESP32 streams sensor CSV and the **laptop** runs the network. The deliverable explicitly requires *"Model optimization for Arduino / edge deployment (quantization, pruning)"* and *"Real-time performance on Arduino (inference time, memory footprint)"* | ❗⬜ |

The professor's verbal hardware component plus deliverable sections 7 and 10 mean the
network has to run **on the board**, with measured inference time and memory.
Plan (see §3, Phase D): quantise a compact CNN to int8, emit it as a C header,
run the forward pass on the ESP32 in C, and prove the on-device implementation is
numerically identical to PyTorch through a host-compiled parity test.

### 2.7 Write-up & presentation

| # | Item | Status |
|---|---|---|
| 33 | `docs/architecture.md` (predecessor) needs rewriting for the merged system | ⬜ |
| 34 | `docs/TASKS_ALIGNMENT.md` — each of the 3 tasks answered with measured numbers | ⬜ |
| 35 | Write-up drafted against the base-paper template, sections 1–11 | ⬜ |
| 36 | Block diagram, circuit diagram, hardware connection diagram, NN architecture diagram | ⬜ |
| 37 | Dataset drive link | ⬜ |
| 38 | Presentation deck + demo script | ⬜ |

---

## 3. Realistic scope and measured cost

### 3.1 What actually costs time (measured, not estimated)

| Model | Wall clock, this machine (CPU) |
|---|---|
| 1-D CNN | **35 s** |
| Patch-ViT | **38 s** |
| CNN + Transformer | 4 min 19 s |
| BiLSTM | ~5 min |
| pure Transformer (480-step attention) | ~45 min with a 15-epoch budget |
| all four classical baselines | ~60 s |

Fact worth stating plainly: **the best-scoring model we have trained in under a
minute.** Compute is not the constraint on this project. The pure Transformer is
the single expensive item, because attention over 480 timesteps is O(T²) — and
that is itself a reportable finding (`results/benchmark.json`).

The **entire** experiment matrix is therefore ~2.5–3 hours of unattended machine
time, run in the background. It does not block the demo, the write-up or the
hardware work.

### 3.2 The strategy that earns the marks (and needs almost no compute)

The base paper reports **99.7 %** from **window-level k-fold cross-validation**,
which places windows from the same person in both train and test. That number is
not comparable to a wearable that must work on a person it has never seen.

So the write-up reports **two protocols** rather than chasing 99.7 %:

| Protocol | Methodology | What it shows |
|---|---|---|
| **Base-paper** | window-level stratified k-fold, subjects mixed across folds | our models measured under *their* protocol — a like-for-like comparison against their SVM/NB/99.7 % |
| **Honest** | subject-disjoint train/val/test | what the device actually achieves on a new wearer, plus the reason the first number is inflated |

This is a stronger result than reproducing 99.7 %, and the base-paper phase costs
about 20 minutes because only the cheap models are needed to make the point.

### 3.3 Deliberately cut (to protect the critical path)

* Hardware-only ablation → baselines + one CNN instead of the full zoo (still
  answers Task 2 with a measured number).
* Binary task → baselines + one deep model instead of the full zoo.
* Pure Transformer → 15 epochs / patience 6 instead of 60, and the limit is
  stated as a resource constraint in the report.

---

## 4. Remaining work, in execution order

### Phase A — finish the model evidence (background, unattended)
1. Configuration sweep (normalisation × augmentation) → `results/sweep_summary.json`.
2. 3-class, full channels, all models → `results/comparison.json`, `*_metrics.json`, confusion matrices.
3. 3-class, **hardware-only** channels → the Task-2 sensor ablation (`comparison_hw.json`).
4. **Binary** stress-vs-baseline, full channels → `comparison_binary.json`.
5. **Base-paper protocol** window k-fold → `results/kfold5.json`.
6. `scripts/plot_results.py` → the comparison figure + `results/model_comparison.md`.
7. ONNX export of every checkpoint with a PyTorch-parity check (`scripts/export_onnx.py --all`).

### Phase B — ⭐⭐ the live end-to-end demo (highest mark value)
8. `python run.py`, then drive it: replay a real subject, switch to hardware mode,
   pull a demo scenario, confirm the FSM walks SAFE→CAUTION→THREAT→ALERT and that
   exactly one alert is dispatched, and that the WebSocket keeps streaming.
9. Prove the vision worker does not throttle telemetry (FPS/CPU numbers, with and
   without the camera) → the F10 evidence.
10. Timed demo script so the 5-minute review is rehearsed, not improvised.

### Phase C — ESP32 firmware verified
11. `arduino-cli compile` the ESP32 sketch; record flash/RAM.
12. Loopback test: the firmware's CSV line parses through `serial_source` unchanged.

### Phase D — ⭐⭐ real on-device inference (the deliverable's edge-deployment item)
13. `scripts/export_edge.py` — fold BatchNorm, quantise weights to int8 with
    per-channel scales, emit `firmware/edge_model/*.h` (weights + biases + scaler
    mean/std + window/channel metadata).
14. `firmware/edge_model/edge_infer.c` — a dependency-free C forward pass for the
    compact CNN, usable both by the ESP32 sketch and (on the host) by a test.
15. `tests/test_edge_parity.py` — compile that C with `cc`, run random windows, and
    assert the outputs match PyTorch within tolerance. **This is the proof the
    "deployed" model is the same model.**
16. ESP32 sketch `esp32_edge_wearable/` — sense → window → normalise → infer on-board
    → OLED + `STATE:`/`SCORE:` over serial/BLE.
17. Measure and record: int8 model flash bytes, SRAM arena, ms per inference, and
    the estimated battery life at the real duty cycle; compare against the base
    paper's 3 s latency and 18–20 h battery.

### Phase E — documentation & the write-up
18. Rewrite `README.md` for the merged project (it still documents the dead
    `fittrack/` layout).
19. `docs/ARCHITECTURE.md` — merged system, with the data contract and the two
    deployment modes (laptop-hosted vs on-device).
20. `docs/TASKS_ALIGNMENT.md` — the 3 tasks, each answered with measured numbers.
21. Write-up in the base-paper template, sections 1–11, one column.
22. Diagrams: block diagram, circuit/connection diagram, NN architecture, pipeline.
23. Presentation outline, or a prompt to generate the deck from the real figures.

### Phase F — acceptance pass
24. Re-run the full test suite, re-verify every generated artifact, and check each
    numbered deliverable requirement against the artefact that satisfies it.

---

## 5. What is genuinely at risk

| Risk | Mitigation |
|---|---|
| Pure Transformer is ~10× slower per batch than CNN+Transformer on CPU (`results/benchmark.json`) | Budgeted to 15 epochs / patience 6 (~45 min, unattended); it runs last so it can never block anything. The limit is stated as a resource constraint in the report. |
| The ESP32 board core download keeps being interrupted | It is only needed for the *compile* of the wireless sketch. The on-device inference evidence (Phase D) is verified with `cc` on the host, which needs no board core at all. |
| On-device training-free deployment needs TensorFlow, which may have no wheel for Python 3.14 | Deliberately avoided: the C export path needs **no** TF, only NumPy — and it is verified by compiling with `cc`. |
| Subject-independent WESAD is hard; macro-F1 will not approach 99.7 % | That is the point. The base paper's number comes from window-level k-fold. Reporting a lower, honest number *plus* the reason is a stronger result than reproducing a leaky 99.7 %. |
| Deadline 15–16/10 | Phases A–C and E are on the critical path; Phase D is the differentiator; Phase F is reserved time. |

---

## 6. Ground rules this project holds itself to

* No fabricated sensor values. A missing reading is `None` / an empty CSV field.
* No leakage: subject-disjoint splits, scaler fitted on train only.
* Every number in the write-up comes from a run on **real** WESAD, with the
  artefact that produced it in `results/`.
* Every fix in `docs/FLAWS_AND_FIXES.md` has a test or a measurement behind it.
* A model is not "deployed" unless it runs on the board and is shown to produce
  the same answers as the reference implementation.
