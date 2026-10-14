# Architecture

One pipeline, several data sources, two inference hosts.

```
                          ┌──────────────────────────────────────────┐
   real WESAD wrist  ────►│                                          │
   ESP32 + MAX30102  ────►│            SensorSource                  │
   ESP32 + MPU6050   ────►│   (one interface, one packet schema)      │
   recorded CSV log  ────►│                                          │
   simulator         ────►│                                          │
                          └───────────────────┬──────────────────────┘
                                              │  SensorPacket
                                              │  (a missing reading is None)
                                              ▼
                          ┌──────────────────────────────────────────┐
                          │            WindowBuffer                  │
                          │  resample to the checkpoint's own fs,    │
                          │  return None until a full window exists  │
                          └───────────────────┬──────────────────────┘
                                              │  (T, C) float32
                       ┌──────────────────────┴───────────────────────┐
                       ▼                                              ▼
        ┌──────────────────────────┐                ┌────────────────────────────┐
        │  HOST-SIDE (laptop)      │                │  ON-DEVICE (ESP32)         │
        │  SensorMLPredictor       │                │  int8 CNN, C forward pass  │
        │  PyTorch sequence model  │                │  (see EDGE_DEPLOYMENT.md)  │
        │  temperature-calibrated  │                │                            │
        └────────────┬─────────────┘                └──────────────┬─────────────┘
                     │  P(stress)                                  │  P(stress)
                     └──────────────────┬──────────────────────────┘
                                        ▼
                        ┌───────────────────────────────┐
   camera ──► VisionWorker ──► visual_distress ──►│        FusionEngine         │
   (own thread, never blocks telemetry)           │  weights + suppression      │
                        └───────────────┬───────────────┘
                                        │  fused threat score [0,1]
                                        ▼
                        ┌───────────────────────────────┐
                        │      AlertStateMachine        │
                        │  SAFE→CAUTION→THREAT→ALERT→   │
                        │  RECOVERY→SAFE, cooldown      │
                        └───────────────┬───────────────┘
                                        │  state transition event
                                        ▼
                        ┌───────────────────────────────┐
                        │        AlertDispatcher        │
                        │  dedup · cooldown · retry ·   │
                        │  audit trail                  │
                        └───────────────┬───────────────┘
                                        ├──► TelegramNotifier   (implemented)
                                        ├──► ConsoleNotifier
                                        └──► NullNotifier      (default)
```

---

## 1. Package layout

```
threatwatch/
├── run.py                      single-command launcher (uvicorn)
├── threatwatch/
│   ├── config.py               the only place constants live
│   ├── data/
│   │   ├── types.py            SensorPacket · SensorSource ABC · WindowBuffer
│   │   ├── wesad.py            real-WESAD wrist loader (1/64 g fix, HR from BVP)
│   │   ├── dataset.py          windowing · Standardizer · leakage-free splits
│   │   ├── augment.py          on-the-fly window augmentation
│   │   ├── replay_engine.py    speed / seek / scenario jump over a real subject
│   │   └── sources/            wesad · serial · ble · log_replay · simulator · hardware
│   ├── models/
│   │   ├── seq_models.py       CNN · BiLSTM · Transformer · CNN+Transformer · PatchViT
│   │   └── baselines.py        SVM · Naive Bayes · RandomForest · LogisticRegression
│   ├── ml/
│   │   ├── training.py         training loop · early stop · temperature calibration · ECE
│   │   ├── metrics.py          accuracy · macro-F1 · per-class · confusion matrix
│   │   ├── inference.py        checkpoint loader · calibrated live inference
│   │   ├── hardware_inference.py  heuristic fallback for a bare board
│   │   └── export.py           ONNX (verified) + the documented TFLite route
│   ├── vision/                 camera · YOLOv8-pose detector · pose features · worker
│   ├── fusion/                 engine (weights + suppression) · state_machine
│   ├── alerts/                 base(Notifier) · telegram · manager(cooldown/dedup/retry)
│   ├── api/                    app (PipelineRunner) · routes · websocket
│   └── utils/logging.py        stdlib logger factory + safety audit log
├── frontend/                   dashboard: live charts, log panel, replay + demo controls
├── firmware/
│   ├── esp32_wearable/         the board that matches the hardware (sensing + BLE)
│   ├── esp32_edge_wearable/    sensing + on-device inference
│   └── arduino_uno_wearable/   sensing only — kept as a size-feasibility comparison
├── scripts/                    download · preprocess · train · experiments · evaluate
│                               · benchmark_models · export_onnx · live · plot_results
├── tests/                      one merged suite (see §6)
└── docs/
```

---

## 2. The data contract

Everything downstream depends on one type:

```python
@dataclass
class SensorPacket:
    timestamp: float
    heart_rate: Optional[float]     # None = not measured
    spo2: Optional[float]
    eda: Optional[float]
    temperature: Optional[float]
    acc_x/acc_y/acc_z: Optional[float]
    ...
    label: int = -1                 # -1 = unknown (always, on live hardware)
    source: str
```

**`None` means "this sensor did not report", and it never becomes a plausible
number.** This is the single most important rule in the codebase. An absent EDA
or SpO₂ reading stays absent all the way into the window builder, which
interpolates only across genuinely valid samples and falls back to the channel
mean. The predecessor project filled missing fields with defaults
(`eda=3.0, heart_rate=75.0`) and invented an SpO₂ value from an uncalibrated
`red/ir` ratio; both are gone.

`WindowBuffer` then:

* stores packets with their real timestamps,
* resamples onto a uniform grid at the **checkpoint's own** `target_fs`,
* returns `None` until a full contiguous window exists.

That last point matters: the predecessor's replay wrapped around at the start of a
recording, concatenating the tail of the session with its head and feeding the
model a window containing a time discontinuity. Today the model simply waits, and
the UI shows "buffering".

---

## 3. Self-describing checkpoints

A checkpoint is not just weights. It carries:

| Field | Why |
|---|---|
| `state_dict`, `model_name`, `model_kwargs` | rebuild the exact network |
| `n_channels`, `channels` (names) | a channel *name* mismatch is an error, not a silent skew |
| `window` (`window_sec`, `stride_sec`, `target_fs`) | the live window is built to match training |
| `scaler` (mean, std, channels, **mode**) | the live path normalises exactly as training did |
| `temperature` | fitted on validation (below) |
| `class_names` | so "stress" is found by name, not by assuming index 1 |

Loading a checkpoint that does not record its window is a hard `ValueError`. That
is deliberate: the predecessor had two projects with two window sizes (128 vs 960
samples) and no record of which belonged to which model.

### Normalisation mode is an architectural decision, not a detail

`mode="global"` z-scores with statistics fitted on the training subjects.
`mode="window"` z-scores each window against itself.

Measured on the honest subject-disjoint test set, this is the difference between
the project working and not working:

| normalisation | test macro-F1 |
|---|---|
| global | 0.330 |
| **per-window** | **0.612** |

A person's resting heart rate, skin temperature and EDA baseline are personal
offsets. Global statistics leave those offsets in place, so a new subject's
windows land off the training manifold. Per-window z-scoring cancels the offset by
construction — and, usefully for the ESP32, it needs **no stored statistics at
all**, so the board can normalise the window it just captured with no dataset
lookup.

---

## 4. Two inference hosts, one model definition

| | Host-side | On-device |
|---|---|---|
| Where | laptop CPU | ESP32, 240 MHz |
| Model | any of the five sequence models | compact 1-D CNN, int8 weights |
| Normalisation | checkpoint scaler | same, recomputed on-board |
| Purpose | accuracy ceiling, the reported benchmark | the deliverable's edge-deployment requirement |
| Proof of equivalence | PyTorch vs ONNX (`onnxruntime`) | **PyTorch vs compiled C** (`tests/test_edge_parity.py`) |

The two paths are not two models. The edge path is the same architecture, trained
separately at a size that fits the board, exported by `scripts/export_edge.py`
and verified against PyTorch by compiling the *same C source* on the host and
comparing outputs numerically. If the C forward pass and PyTorch disagree, the
test fails — a model is not "deployed" because a file was copied to a board.

---

## 5. Fusion and alerting

`FusionEngine` combines the wearable distress signal and the vision distress
signal with configurable weights, then applies documented suppression rules —
for example, a high sensor score with a calm scene is capped, and a high vision
score with an unremarkable physiology is capped. Every threshold lives in
`config.py`; the predecessor had them as inline literals duplicated from its
config.

`AlertStateMachine` is an explicit five-state machine with per-transition timer
resets and a monotonic clock. The predecessor had three latent bugs here (a
threat timer never cleared on entry, a recovery timer not cleared on exit, and a
SAFE→CAUTION transition that only fired on the *second* over-threshold sample),
plus an `ALERT ACTIVE` state that auto-advanced 1.5 s after THREAT regardless of
whether the score had actually fallen.

`AlertDispatcher` is what turns a state transition into a message that reaches a
human: de-duplication by signature, a cooldown window, retry with backoff, and an
audit record. The predecessor's "emergency dispatch" set a boolean to `True` and
returned the string `"simulated emergency payload dispatched"` — nothing was ever
sent.

---

## 6. Threading and the event loop

The FastAPI event loop owns: the WebSocket broadcast at `WEBSOCKET_FPS`, the
fusion/FSM step, and the alert dispatch.

**Vision does not run on it.** `VisionWorker` is a dedicated thread doing
capture → YOLOv8-pose → pose analysis → overlay, publishing an immutable
`VisionSnapshot` that the loop only reads. In the predecessor, `detect()` was
called inline in the same async loop responsible for telemetry, so a slow frame
stalled every connected client — the dashboard stuttered exactly when the scene
was interesting enough to matter.

Alert dispatch is likewise off the hot path (`dispatch_sync` via
`asyncio.to_thread`), and notification failure is non-fatal by construction: a
dead Telegram bot cannot take down the pipeline.

---

## 7. Tests

| File | Covers |
|---|---|
| `test_wesad_loader.py` | real loader: output length, label mapping, **1/64 g conversion**, HR from a 1.2 Hz pulse |
| `test_pipeline.py` | windowing, majority labelling, transient rejection, Standardizer, both normalisation modes |
| `test_serial_loopback.py` | full hardware path over a **pseudo-terminal** — the firmware's exact CSV lines, parsed by the real `SerialSensorSource`, into a `WindowBuffer` |
| `test_replay_engine.py` | honest `None` fields, time-accurate 1×/4× playback, fractional-tick drift, scenario seeking, end wrap |
| `test_fusion.py` | weighting, suppression rules, clamping |
| `test_state_machine.py` | every transition, timer resets, cooldown suppression |
| `test_alerts.py` | one message per escalation, dedup, cooldown, failure isolation |
| `test_edge_parity.py` | C forward pass vs PyTorch on the same windows |

The serial test is the interesting one: it stands up a pseudo-terminal and has a
"device" thread write the firmware's literal output, so the hardware path is
exercised end-to-end with no board attached. That is why the firmware protocol and
`serial_source.py` cannot drift apart unnoticed.

---

## 8. What is deliberately *not* here

* **No fabricated sensor data.** The predecessor's "WESAD" files were three
  identical 27 938 584-byte synthetic pickles presented as the real dataset. This
  project ships the real 15-subject WESAD and every reported number comes from it.
* **No GNN.** Task 1 lists Graph Neural Networks. A 7-channel sensor graph is
  tiny and not meaningfully structured, so a GNN would be a decoration rather than
  a model. The write-up explains the reasoning and what would make a GNN worth
  adding (a multi-device/multi-person body-sensor graph) instead of bolting one on.
* **No uncalibrated SpO₂.** Emitted empty, with a comment stating what calibration
  would require.
* **No auto-escalation.** `ALERT ACTIVE` requires the score to still be in the
  THREAT band; it does not advance on a timer alone.
