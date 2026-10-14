"""
Single source of truth for the ThreatWatch project.

Everything that used to be duplicated across two projects (paths, subject
splits, channel lists, window sizes, thresholds, sampling rates) lives here and
nowhere else.

Runtime secrets / deployment overrides are read from the environment (and from a
``.env`` file if present) — see :class:`Settings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: only needed for .env convenience
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    pass


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
WESAD_DIR = DATA_DIR / "wesad"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
FRONTEND_DIR = BASE_DIR / "frontend"
FIGURES_DIR = BASE_DIR / "docs" / "figures"

for _p in (DATA_DIR, WESAD_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Dataset: WESAD
# --------------------------------------------------------------------------- #
# WESAD ships subjects S2..S17.  S1 does not exist; S12 has known sensor
# problems and is excluded by convention.
ALL_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]  # 15 subjects

# Subject-independent split.  A subject appears in exactly one split, always.
# This is the single most important guard against the leakage bug that plagued
# the predecessor project.
TRAIN_SUBJECTS = ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11"]
VAL_SUBJECTS = ["S13", "S14"]
TEST_SUBJECTS = ["S15", "S16", "S17"]

# Native sampling rates (Hz) of the WESAD wrist device (an Empatica E4).
WRIST_RATES = {"ACC": 32.0, "BVP": 64.0, "EDA": 4.0, "TEMP": 4.0}
CHEST_FS = 700.0

# Empatica E4 reports wrist accelerometer counts in units of 1/64 g.
WRIST_ACC_LSB = 64.0

# Common rate every channel is resampled to before windowing.
TARGET_FS = 32.0

# WESAD label codes -> our class indices.  0/4/5/6/7 are transient/ignore.
LABEL_MAP = {1: 0, 2: 1, 3: 2}
CLASS_NAMES = ["baseline", "stress", "amusement"]
NUM_CLASSES = 3
BINARY_CLASS_NAMES = ["baseline", "stress"]
NUM_BINARY_CLASSES = 2


# --------------------------------------------------------------------------- #
# Channels
# --------------------------------------------------------------------------- #
# Canonical order of the *packet* vector.  ``WindowBuffer`` stores samples in
# this order; every channel list below is a subset of it, so there is exactly
# one place that knows how a packet becomes a row of numbers.
PACKET_CHANNELS = ["hr", "eda", "temp", "acc_x", "acc_y", "acc_z", "acc_mag"]

# What WESAD + our hardware can jointly provide (the "honest" set).
FEATURE_CHANNELS = ["hr", "eda", "temp", "acc_x", "acc_y", "acc_z", "acc_mag"]

# Hardware-only subset: the current BOM (MAX30102 + MPU6050) has no GSR/EDA
# sensor, so this is what the physical prototype can actually reproduce.
HW_FEATURE_CHANNELS = ["hr", "temp", "acc_x", "acc_y", "acc_z", "acc_mag"]

CHANNEL_SETS = {"full": FEATURE_CHANNELS, "hw": HW_FEATURE_CHANNELS}


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window parameters.  Frozen so it cannot drift mid-run."""

    window_sec: float = 15.0
    stride_sec: float = 5.0
    target_fs: float = TARGET_FS

    @property
    def window_samples(self) -> int:
        return int(round(self.window_sec * self.target_fs))

    @property
    def stride_samples(self) -> int:
        return int(round(self.stride_sec * self.target_fs))

    def tag(self, channels_key: str = "full") -> str:
        suffix = "" if channels_key == "full" else f"_{channels_key}"
        return f"w{int(self.window_sec)}_s{int(self.stride_sec)}_fs{int(self.target_fs)}{suffix}"


DEFAULT_WINDOW = WindowConfig()


# --------------------------------------------------------------------------- #
# Training defaults
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.10        # fraction of epochs used for LR warmup
    patience: int = 12               # early stopping on val macro-F1
    device: str = "auto"             # auto | cpu | cuda
    num_workers: int = 0             # 0 is safest on the /mnt/c 9p mount
    label_smoothing: float = 0.05
    class_weights: bool = True       # inverse-frequency weighting
    max_train_windows: int = 0       # 0 = no cap
    grad_clip: float = 5.0
    tensorboard: bool = True
    augment: bool = True              # on-the-fly window augmentation (see data/augment.py)
    normalize: str = "global"         # "global" | "window" — see data/dataset.py


# Per-model batch sizes.  Attention over T timesteps costs O(T^2) memory, so
# transformer variants get a smaller batch.
AUTO_BATCH = {"cnn": 64, "bilstm": 64, "cnn_transformer": 64, "transformer": 16, "patch_vit": 32}


# --------------------------------------------------------------------------- #
# Vision subsystem
# --------------------------------------------------------------------------- #
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov8n-pose.pt")
POSE_CONF_THRESHOLD = 0.40

# Pose heuristic thresholds (all normalised, see vision/pose_features.py)
FALL_SHOULDER_TILT_DEG = 50.0
FALL_ASPECT_RATIO = 0.65
DEFENSIVE_WRIST_HEAD_PX = 35.0
DEFENSIVE_WRIST_LATERAL_FRAC = 0.20
RAPID_MOTION_THRESH = 2.2
MODERATE_MOTION_THRESH = 0.9
SUPPRESS_VISION_CAP = 0.46
VISION_AGREEMENT_LOW = 0.30
VISION_AGREEMENT_HIGH = 0.60


# --------------------------------------------------------------------------- #
# Multimodal fusion
# --------------------------------------------------------------------------- #
SENSOR_WEIGHT = 0.60
VISION_WEIGHT = 0.40

SAFE_UPPER_BOUND = 0.39
CAUTION_UPPER_BOUND = 0.69
THREAT_LOWER_BOUND = 0.70

# Mutual-agreement escalation
AGREEMENT_SENSOR_MIN = 0.65
AGREEMENT_VISION_MIN = 0.60
AGREEMENT_BOOST = 1.12
AGREEMENT_FLOOR = 0.78

# Single-modality false-alarm suppression
SUPPRESS_SENSOR_MIN = 0.60
SUPPRESS_VISION_MAX = 0.30
SUPPRESS_SENSOR_CAP = 0.50
SUPPRESS_VISION_MIN = 0.60
SUPPRESS_SENSOR_MAX = 0.30

ROLLING_WINDOW_SIZE = 5       # inference samples averaged before the FSM sees a score


# --------------------------------------------------------------------------- #
# Alert state machine
# --------------------------------------------------------------------------- #
THREAT_CONSECUTIVE_SECS = 2.5   # score must stay in THREAT band this long
RECOVERY_CONSECUTIVE_SECS = 3.0  # score must stay low this long to recover
ALERT_ESCALATE_SECS = 1.5        # THREAT -> ALERT ACTIVE delay
ALERT_COOLDOWN_SECS = 30.0       # minimum gap between two dispatched alerts


# --------------------------------------------------------------------------- #
# Alerts / notifications
# --------------------------------------------------------------------------- #
@dataclass
class AlertSettings:
    enabled: bool = os.getenv("ALERTS_ENABLED", "0") == "1"
    channel: str = os.getenv("ALERT_CHANNEL", "telegram")  # telegram | console
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    cooldown_secs: float = float(os.getenv("ALERT_COOLDOWN_SECS", str(ALERT_COOLDOWN_SECS)))
    timeout_secs: float = float(os.getenv("ALERT_TIMEOUT_SECS", "8"))
    max_retries: int = int(os.getenv("ALERT_MAX_RETRIES", "2"))
    include_frame: bool = os.getenv("ALERT_INCLUDE_FRAME", "0") == "1"

    @property
    def telegram_ready(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


# --------------------------------------------------------------------------- #
# Hardware link
# --------------------------------------------------------------------------- #
DEFAULT_SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")
DEFAULT_SERIAL_BAUD = int(os.getenv("SERIAL_BAUD", "115200"))
HARDWARE_SAMPLING_RATE_HZ = 15.0

# Downstream (laptop -> device) protocol.  Two simple lines keep the ATmega328P
# parser tiny and preserve the original ``STATE:<name>`` contract.
DEVICE_STATE_PREFIX = "STATE:"
DEVICE_SCORE_PREFIX = "SCORE:"

# Wireless link: ESP32 Nordic UART Service over BLE.  The firmware
# (firmware/esp32_wearable) is the source of truth for these UUIDs; the same
# values appear in scripts/windows_ble_bridge.py.
BLE_DEVICE_NAME = os.getenv("BLE_DEVICE_NAME", "ThreatWatch")
BLE_NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
BLE_NUS_RX_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # laptop -> device
BLE_NUS_TX_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # device -> laptop
BLE_PORT_ALIAS = "BLE"  # select the wireless transport by "port" name


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
WEBSOCKET_FPS = int(os.getenv("WEBSOCKET_FPS", "20"))


# --------------------------------------------------------------------------- #
# Checkpoint helpers — a checkpoint always carries its own window config,
# channel list and scaler, so live inference cannot silently mismatch training.
# --------------------------------------------------------------------------- #
SCALER_FILENAME = "scaler.json"
LATEST_DATASET_FILENAME = "latest_dataset.json"


def settings() -> AlertSettings:
    """Return alert settings freshly read from the environment."""
    return AlertSettings()
