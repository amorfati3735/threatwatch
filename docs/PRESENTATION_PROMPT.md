# Prompt for generating the presentation deck

Paste everything in the block below into the web AI you are using for slides.
It is written to produce a **10-slide deck for a 5-minute review**, using only
numbers that exist in this repository.

Before pasting, replace the two `<...>` placeholders with the real values from
`results/` (`docs/WRITEUP.md §10` has the current table).

---

```
Create a 10-slide presentation deck for a university "Embedded Systems" capstone
project review. It must be readable in 5 minutes, so keep each slide to a title, a
short subtitle, at most 4 bullets of ≤12 words, and one visual.

CRITICAL RULES
- Use ONLY the numbers provided below. Do not invent, round up, or "improve" any
  metric. If a figure is missing, write "measurement pending" rather than a guess.
- Do not add slides about technologies that are not in the project.
- Do not claim the project beats the reference paper's 99.7% accuracy. The project's
  contribution is showing that 99.7% is an artefact of the evaluation method.
- Suggest a visual for each slide (diagram, chart, photo, table). Where a photo is
  suggested, say what should be photographed.

PROJECT TITLE
"IoT-Based Multimodal Distress Detection on an ESP32 Wearable with Edge-Deployed
Deep Learning"

AUDIENCE
Engineering professors. They care about: correctness, engineering justification,
measured results, honest limitations, and a working demo. They will not read the
paper during the review.

CONTENT (use exactly these facts)

Slide 1 - Title
  Title, your name, course, date.
  Subtitle: "Replacing SVM/Naive-Bayes with deep learning - and measuring what that
  actually buys, on real data."
  Visual: photo of the assembled wearable (ESP32 + MAX30102 + MPU6050 + OLED on
  breadboard) held in a hand.

Slide 2 - The problem
  - Existing personal-safety devices need the wearer to press a button.
  - That is unrealistic during an incident: the person may be restrained or unable
    to reach the device.
  - So the detector must be autonomous: infer distress from physiology, no user
    action.
  - Base paper: Wagh et al., Scientific Reports 2025, reports 99.7% accuracy using
    SVM and Naive Bayes.
  Visual: a simple 2-box diagram - "manual activation (fails)" vs "autonomous
  inference (this project)".

Slide 3 - What was built
  Four subsystems:
  - Wearable: ESP32 + MAX30102 (PPG -> heart rate) + MPU6050 (accelerometer/gyro)
    + SSD1306 OLED, all on one I2C bus.
  - Vision: camera running YOLOv8-pose, interpreting skeleton keypoints for falls,
    defensive posture and violent motion.
  - Fusion + alerts: weighted fusion, a 5-state machine (SAFE -> CAUTION -> THREAT
    -> ALERT ACTIVE -> RECOVERY), and a real notifier with de-duplication,
    cooldown and retry.
  - On-device inference: a quantised int8 neural network running ON the ESP32.
  Visual: the system block diagram from docs/ARCHITECTURE.md (redraw it cleanly:
  sensors -> ESP32 -> host -> vision + fusion -> alert).

Slide 4 - Data: real, and honestly split
  - Dataset: WESAD, real, 15 participants (S2-S17).
  - Trained ONLY on wrist signals the hardware can actually produce: heart rate
    from PPG, EDA, temperature, accelerometer.
  - Participants split BEFORE windowing: train S2-S11, validate S13-S14, test
    S15-S17.
  - No participant appears in two splits. No leakage.
  Visual: a horizontal bar showing the 15 participants partitioned into three
  disjoint blocks, train/val/test, with the counts 4354 / 880 / 1333 windows.

Slide 5 - The evaluation trap  (THIS IS THE KEY SLIDE)
  - The base paper's 99.7% comes from window-level k-fold: windows from the SAME
    person land in both training and test.
  - Three independent recent papers on WESAD report 98%, 99.7% and 99.73-99.95%.
    Different models, different sensors, same narrow band - a shared methodological
    cause.
  - For a wearable that must work on a stranger, that is not a real accuracy.
  - So we measured BOTH protocols on the same pipeline.
  Visual: two-bar chart, "same models, same data": base-paper protocol (high, label
  "comparable to published work") vs subject-disjoint protocol (lower, label "what
  a new wearer actually gets"). Use the real k-fold number from results/kfold5.json
  when available.

Slide 6 - Result 1: normalisation mattered more than architecture
  - Per-window normalisation cancels each person's personal physiological offset.
  - 1-D CNN: 0.330 -> 0.617 macro-F1. Patch-ViT: 0.331 -> 0.586.
  - That single change is worth MORE than the gap between any two architectures.
  - It also collapsed the validation-to-test gap from +0.325 to +0.007.
  Visual: grouped bar chart, macro-F1, CNN and Patch-ViT each shown with "global"
  vs "per-window" bars. Insert the validation-test gap as a text annotation.

Slide 7 - Result 2: the deep model beats SVM, and SVM was actually re-run
  - We ran SVM and Naive Bayes ourselves, not just cited them.
  - Test macro-F1 (subject-disjoint): 1-D CNN 0.617, SVM 0.486, Naive Bayes 0.375,
    logistic regression 0.496.
  - So the CNN beats the base paper's own classifier by +0.131 macro-F1.
  - Caveat: with the naive global normalisation the CNN LOSES to SVM (0.330 vs
    0.486). The win comes from the data handling, not from using a bigger model.
  Visual: horizontal bar chart, models ranked by test macro-F1, with SVM and Naive
  Bayes highlighted in a different colour and labelled "base paper's classifiers".

Slide 8 - Result 3: the model is edge-ready, parity-proven
  - int8 weights with per-output-channel scales; batch-norm folded into the
    convolutions; activations stay float32 (the ESP32 has a hardware FPU).
  - Verified numerically: the C forward pass is compiled on the host and compared
    against PyTorch - bit-level identical (max logit diff 0.00000).
  - int8 costs nothing: identical predictions on real test windows (accuracy
    0.7075 -> 0.7075).
  - Firmware footprint measured: 1,123,215 bytes flash (85%), 284,796 bytes SRAM
    free. On-board inference build pending: the activation arena moves to heap.
  Visual: a table of the measurements, plus a photo of the wearable hardware.
  If the on-device demo is not ready, show the parity-test terminal output as
  the proof artefact instead of an OLED photo.

Slide 9 - Honesty and limitations (be direct, this earns credit)
  - The kit has no GSR/EDA sensor, so the strongest stress feature is missing; we
    MEASURE that loss by training a second model without it instead of imputing it.
  - SpO2 is deliberately reported as EMPTY: without a per-sensor calibration curve
    any value would be fabricated.
  - A missing sensor reading is never replaced with a plausible number.
  - Only 15 participants; laboratory-induced stress, not real-world threat.
  - No GNN: a 7-channel sensor graph has no meaningful topology, so it would be
    decoration. Explained rather than bolted on.
  Visual: two simple icons side by side, "what we measured" vs "what we refuse to
  fake".

Slide 10 - Demo and conclusion
  - Wear the band, stand in front of the camera, trigger the stress scenario.
  - Watch: SAFE -> CAUTION -> THREAT -> ALERT ACTIVE on the dashboard, and the
    alert actually dispatch.
  - Conclusion: published 98-99.95% figures on this data are largely an evaluation
    artefact. The dominant variable here was how the data was normalised, not how
    large the model was.
  Visual: a screenshot of the live dashboard during an alert, plus a 3-step demo
  storyboard (1. wear it, 2. stress it, 3. alert fires).

STYLE
- Clean, technical, high contrast. Dark slide background suits the dashboard
  screenshots; keep body text white or near-white.
- No clip-art. No stock photos of people. Use real photos of the actual device and
  real screenshots of the actual dashboard.
- Sans-serif, generous sizes - this will be projected and read from the back of a
  room.
- One idea per slide.

OUTPUT
Give me the deck slide by slide, with speaker notes under each slide written in
first person, timed so the whole thing fits in 5 minutes (about 25-30 seconds of
speaking per slide). Flag any slide where I need to paste in a real figure that I
have not given you.
```

---

## What to fill in before pasting

| Placeholder | Where to get it |
|---|---|
| the k-fold number (slide 5) | `results/kfold5.json`, field `summary.<model>.macro_f1_mean` |
| inference latency (slide 8) | host C reference: `pytest tests/test_edge_parity.py -s` prints 5.8 ms/window (x86). **On-board number pending** — write "measurement pending" in the deck rather than guessing. |
| the hardware-ablation number (slide 9) | `results/comparison_hw.json` for `cnn` vs `cnn` on the full channel set |

## Where the real figures already exist

| Figure | File |
|---|---|
| sweep bar chart data | `results/sweep_summary.json` |
| model comparison table | `results/model_comparison.md` |
| per-model confusion matrices | `results/<model>_confusion.png` |
| architecture / block diagram | `docs/ARCHITECTURE.md` (redraw from the ASCII block diagram) |
| wiring diagram | `docs/HARDWARE_SETUP.md §1` |
| full results discussion | `docs/WRITEUP.md §10` |
