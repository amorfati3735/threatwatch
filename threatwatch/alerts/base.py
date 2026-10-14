"""
Alert delivery primitives.

A :class:`Notifier` is anything that can deliver an :class:`AlertMessage`.  The
interface is deliberately tiny so adding SMS/WhatsApp/webhook later is a small,
self-contained change rather than a refactor.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AlertMessage:
    """A single alert to deliver."""

    title: str
    body: str
    level: str = "EMERGENCY"          # INFO | WARNING | CRITICAL | EMERGENCY
    state: str = "ALERT ACTIVE"
    threat_score: float = 0.0
    source: str = "unknown"           # which sensor source was live
    timestamp: float = field(default_factory=time.time)
    image_path: Optional[Path] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def to_text(self, prefix: str = "[ThreatWatch]") -> str:
        lines = [
            f"{prefix} {self.level} — {self.title}",
            "",
            self.body,
            "",
            f"State          : {self.state}",
            f"Threat score   : {self.threat_score * 100:.1f}%",
            f"Sensor source  : {self.source}",
            f"Detected at    : {self.when}",
        ]
        return "\n".join(lines)


class Notifier(abc.ABC):
    """Delivery backend for alert messages."""

    name: str = "notifier"

    @abc.abstractmethod
    def send(self, message: AlertMessage) -> bool:
        """Attempt delivery.  Must never raise; returns success."""

    def is_configured(self) -> bool:
        """True when the backend has everything it needs to attempt delivery."""
        return True

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "configured": self.is_configured()}


class NullNotifier(Notifier):
    """Does nothing.  Used when alerts are disabled, so the pipeline code path
    stays identical whether or not notifications are turned on."""

    name = "null"

    def send(self, message: AlertMessage) -> bool:
        return False


class ConsoleNotifier(Notifier):
    """Prints the alert to stdout — the zero-configuration fallback."""

    name = "console"

    def __init__(self, prefix: str = "[ThreatWatch]"):
        self.prefix = prefix
        self.sent = 0

    def send(self, message: AlertMessage) -> bool:
        print("\n" + "=" * 62)
        print(message.to_text(self.prefix))
        print("=" * 62 + "\n")
        self.sent += 1
        return True
