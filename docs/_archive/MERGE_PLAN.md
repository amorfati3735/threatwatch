# Merge plan — one project from two

**Goal.** A single, coherent repository that (a) keeps the honest real-WESAD machine
learning from *threat detection project*, (b) keeps the live multimodal dashboard from
*esa project*, (c) fixes the flaws listed in `FLAWS_AND_FIXES.md`, and (d) adds a real
alert-dispatch path so "threat detected" actually reaches a human.

**Constraint.** Nothing is deleted from the two originals until the merged project is
verified. The originals remain at their paths as read-only reference.

---

## 1. Where things came from

| Layer | Source | Why |
|---|---|---|
| WESAD loader (wrist, 1/64 g fix) | A (`fittrack/data/wesad.py`) | only correct implementation |
| Windowing / Standardizer / splits | A (`fittrack/data/dataset.py`) | leakage-free by construction |
| Sequence models | A (`fittrack/models/seq_models.py`) | CNN / BiLSTM / Transformer / CNN+Transformer |
| Classical baselines | A (`fittrack/models/baselines.py`) | honest "how hard is this?" reference |
| Training loop, metrics, plots | A (`fittrack/training.py`, `eval/metrics.py`) | working, early stopping + latency |
| Sensor-source abstraction | A (`fittrack/sources/base.py`) | `SensorPacket` with `None` semantics |
| Serial / BLE / simulator sources | A (`fittrack/sources/*`) | CSV protocol, already tested |
| FastAPI + WebSocket server | B (`backend/main.py`, `api/*`) | the live system |
| Vision (YOLOv8-pose + pose features) | B (`backend/vision/*`) | working detector + HUD |
| Fusion engine + alert state machine | B (`backend/fusion/*`) | the multimodal decision logic |
| Cyber dashboard | B (`frontend/*`) | the demo centrepiece |
| Firmware | A's Uno (compile-verified) + B's OLED HUD | best of both |

---

## 2. Target layout

```
threatwatch/
├── run.py                     # single-command launcher (uvicorn)
├── requirements.txt
├── .env.example
├── threatwatch/               # ONE python package
│   ├── config.py              # single source of truth (paths, splits, thresholds, channels)
│   ├── data/
│   │   ├── types.py           # SensorPacket (None semantics) · SensorSource ABC · WindowBuffer
│   │   ├── wesad.py           # real WESAD wrist loader  [A]
│   │   ├── dataset.py         # windowing · Standardizer · subject splits  [A]
│   │   ├── replay_engine.py   # speed / seek / scenario jump  [B, rewritten]
│   │   └── sources/           # wesad · serial · ble · log_replay · simulator · hardware
│   ├── models/                # seq_models (+PatViT) · baselines · registry
│   ├── ml/                    # metrics · training(+TensorBoard) · inference(+calibration)
│   │                          #   · hardware_inference · export(ONNX)
│   ├── vision/                # camera · detector · pose_features · worker(threaded)
│   ├── fusion/                # engine · state_machine
│   ├── alerts/                # base(Notifier) · telegram · manager(cooldown/dedup/retry)
│   ├── api/                   # app(pipeline runner) · routes · websocket
│   └── utils/logging.py       # audit event log (JSON/CSV export)
├── frontend/                  # cyber dashboard  [B]
├── firmware/                  # arduino_uno/ · esp32/   (one canonical CSV protocol)
├── scripts/                   # download · preprocess · train · evaluate · calibrate
│                              #   · export_onnx · live · replay · plot_results
├── tests/                     # pipeline · wesad_loader · serial_loopback
│                              #   · fusion · state_machine · alerts · replay
├── data/wesad/                # REAL WESAD, 15 subjects (already present)
├── models/  results/  logs/  docs/
```

---

## 3. Data contract (the thing that makes the merge possible)

```
   WESAD wrist replay ─┐
   live Arduino/ESP32 ─┤
   recorded CSV log  ──┼──►  SensorSource ──►  WindowBuffer ──►  model ──► state
   simulator         ──┘      (SensorPacket)   (resample 32 Hz)
```

* Every source yields the **same** `SensorPacket`; a missing reading is `None`.
* `WindowBuffer` resamples onto the checkpoint's own `target_fs`/window length, so the
  live window always matches the training window (fixes F14).
* The window length, channel list and scaler are stored **inside the checkpoint**; loading
  a model that does not match the incoming channels is a hard error, not a silent skew.

---

## 4. Alert path (new capability)

```
fused score ──► AlertStateMachine ──► transition to ALERT ACTIVE
                        │
                        ▼
                 AlertDispatcher  ──► cooldown + dedup + retry + audit
                        │
                        ├──► TelegramNotifier      (implemented)
                        └──► (extensible: SMS/WhatsApp/webhook)
```

Telegram is chosen because it is free, needs no paid account or trial credit, and works
from WSL with a single HTTPS POST. The `Notifier` interface keeps SMS/WhatsApp a
~40-line addition if the write-up wants them.

---

## 5. Phases and gates

| Phase | Deliverable | Gate (must pass before moving on) |
|---|---|---|
| **0** | restructure, plan docs | tree matches §2; no import of `fittrack`/`backend` remains |
| **1** | `config.py`, `data/types.py` | packet round-trips `None`; `WindowBuffer` shape test passes |
| **2** | real-WESAD loader + dataset | loader test passes (length, HR≈72, ±2 g); splits disjoint |
| **3** | replay engine + hardware sources | replay returns `None` until full window; serial CSV round-trip |
| **4** | models + training + metrics | deep models train end-to-end on CPU; JSON metrics written |
| **5** | vision worker | pipeline FPS stays high with vision enabled; snapshot is consistent |
| **6** | fusion + FSM | FSM walks full cycle; fusion suppression rules unit-tested |
| **7** | alerts | one message per escalation; suppressed inside cooldown; failure non-fatal |
| **8** | API + frontend | `python run.py` serves dashboard; replay + demo scenarios work |
| **9** | firmware | Uno sketch compiles under `arduino-cli`; protocol matches `serial_source` |
| **10** | retrain + benchmarks | `results/comparison.json` (3-class) and `comparison_binary.json` from **real** WESAD |
| **11** | docs | `ARCHITECTURE.md`, `TASKS_ALIGNMENT.md`, `PRESENTATION.md` |

---

## 6. Task alignment (the three original assignments)

* **Task 1 — replace SVM/NB with Transformer / LSTM / Bi-LSTM / ViT / GNN and compare.**
  Delivered as a measured comparison on real WESAD with a subject-independent split:
  logistic, random-forest, CNN, BiLSTM, Transformer, CNN+Transformer, and a **patch-based
  ViT** for time series. GNN applicability is analysed (17-channel sensor graph is too
  small/weakly structured) rather than bolted on dishonestly.
* **Task 2 — additional sensors (ECG/PPG/GSR/SpO₂/respiration/EMG/EEG).**
  Delivered as a measurable **sensor ablation**: `full` WESAD channel set vs `hw`
  hardware-only set, quantifying exactly what the missing GSR/EDA module costs, plus a
  documented sensor roadmap (GSR → SpO₂ calibration → respiration).
* **Task 3 — combine wearable sensing with computer vision (YOLOv11/12, RT-DETR, ViT).**
  Delivered as the live multimodal fusion of WESAD/Arduino physiology with YOLOv8-pose
  vision, plus a written analysis of the RT-DETR / pose-sequence-ViT upgrade path and
  why the current heuristic pose analyser is preferred for a deterministic demo.

Details and the exact numbers live in `docs/TASKS_ALIGNMENT.md` once training completes.
