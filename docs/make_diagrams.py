#!/usr/bin/env python3
"""
Generate report-ready block diagrams (PNG) with matplotlib.

    python docs/make_diagrams.py
produces docs/figures/system_blocks.png and docs/figures/pipeline.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures"


def box(ax, x, y, w, h, text, color="#dbeafe", fontsize=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.2, edgecolor="#1e3a8a", facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2, color="#334155"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.4, color=color))


def system_blocks():
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
    ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")

    box(ax, 0.4, 7.4, 4.2, 1.3, "Arduino Uno / ESP32\n+ MAX30102 + MPU6050 + OLED", "#c7f9cc")
    box(ax, 5.4, 7.4, 4.2, 1.3, "WESAD dataset\n(wrist BVP/EDA/TEMP/ACC)", "#fde68a")
    box(ax, 2.4, 5.4, 5.2, 1.1, "SensorSource abstraction\n(serial · BLE · WESAD · replay · simulator)")
    box(ax, 2.6, 3.9, 4.8, 1.0, "WindowBuffer + Standardizer\n(32 Hz, sliding window)")
    box(ax, 2.4, 2.3, 5.2, 1.1, "ML model\nCNN / BiLSTM / Transformer / CNN+Transformer")
    box(ax, 2.9, 0.9, 4.2, 1.0, "Predicted state\nbaseline · stress · amusement")

    arrow(ax, 2.5, 7.4, 4.2, 6.5); arrow(ax, 7.5, 7.4, 5.8, 6.5)
    arrow(ax, 5.0, 5.4, 5.0, 4.9); arrow(ax, 5.0, 3.9, 5.0, 3.4)
    arrow(ax, 5.0, 2.3, 5.0, 1.9)
    ax.text(8.0, 1.4, "→ OLED / dashboard", fontsize=8, color="#334155")
    arrow(ax, 7.1, 1.4, 9.0, 1.4)

    ax.set_title("Fitness Tracker — system block diagram", fontsize=11, fontweight="bold")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "system_blocks.png"); plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(8.5, 2.6), dpi=150)
    ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")
    steps = ["Sensors", "MCU\n(Uno/ESP32)", "timestamped\nstream", "laptop\nreceiver",
             "preprocess\n+ window", "ML model", "predicted\nstate"]
    colors = ["#c7f9cc", "#bae6fd", "#e9d5ff", "#fde68a", "#bbf7d0", "#fecaca", "#c7d2fe"]
    w, gap = 1.6, 0.25
    x = 0.2
    for i, (s, c) in enumerate(zip(steps, colors)):
        box(ax, x, 1.0, w, 1.0, s, c, fontsize=8)
        if i < len(steps) - 1:
            arrow(ax, x + w, 1.5, x + w + gap, 1.5)
        x += w + gap
    ax.set_title("Physical hardware data pipeline", fontsize=11, fontweight="bold")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "pipeline.png"); plt.close(fig)


if __name__ == "__main__":
    system_blocks()
    pipeline()
    print(f"wrote diagrams to {OUT}")
