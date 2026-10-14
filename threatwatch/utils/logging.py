"""
Logging: a stdlib logger factory plus the in-memory safety-event audit log.

Two facilities, deliberately separate:

* :func:`get_logger` / :func:`configure_logging` — ordinary stdlib ``logging`` for
  developer diagnostics.  Level is controlled by ``THREATWATCH_LOG_LEVEL``.
* :class:`SystemEventLogger` — a bounded, timestamped audit trail of *safety*
  events (state transitions, alerts, operator actions) that the dashboard reads
  and the operator can export to JSON/CSV.

Importing this module has **no side effects**.  The audit singleton is created on
first use via :func:`get_event_logger`; ``event_logger`` is a lazy proxy so
existing ``from threatwatch.utils.logging import event_logger`` call sites keep
working without paying an import-time print.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from threatwatch.config import LOGS_DIR

# --------------------------------------------------------------------------- #
# Developer logging
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging once.  Safe to call repeatedly."""
    global _configured
    if _configured and level is None:
        return
    lvl = (level or os.getenv("THREATWATCH_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(level=getattr(logging, lvl, logging.INFO), format=_LOG_FORMAT)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.  Replaces bare ``print`` in scripts."""
    configure_logging()
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Safety-event audit log
# --------------------------------------------------------------------------- #
class SystemEventLogger:
    """
    Centralized event logger for safety alerts, state transitions and telemetry.

    Bounded in memory (``max_entries``) so a long demo cannot exhaust RAM, and
    optionally mirrored to ``logs/events.jsonl`` so the trail survives a restart.
    """

    def __init__(self, max_entries: int = 200, persist: bool = True):
        self.max_entries = max_entries
        self.persist = persist
        self.logs: List[Dict[str, Any]] = []
        self._path = LOGS_DIR / "events.jsonl"
        self.log_event("SYSTEM", "ThreatWatch safety event log initialised.")

    def log_event(
        self,
        category: str,
        message: str,
        level: str = "INFO",
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an event with a formatted timestamp."""
        now = datetime.now()
        entry = {
            "id": len(self.logs) + 1,
            "timestamp_str": now.strftime("%H:%M:%S"),
            "timestamp_iso": now.isoformat(),
            "epoch": time.time(),
            "category": category.upper(),
            "level": level.upper(),
            "message": message,
            "meta": meta or {},
        }
        self.logs.append(entry)
        if len(self.logs) > self.max_entries:
            self.logs.pop(0)

        if self.persist:
            try:
                LOGS_DIR.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except OSError:
                # An unwritable log directory must never break the pipeline.
                self.persist = False

        get_logger(f"threatwatch.{entry['category'].lower()}").log(
            getattr(logging, entry["level"], logging.INFO), message
        )
        return entry

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.logs[-limit:]

    def export_json(self) -> Path:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        path = LOGS_DIR / f"event_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(self.logs, indent=2, default=str), encoding="utf-8")
        return path

    def export_csv(self) -> Path:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        path = LOGS_DIR / f"event_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        fields = ["id", "timestamp_iso", "category", "level", "message", "meta"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in self.logs:
                writer.writerow({
                    "id": row["id"],
                    "timestamp_iso": row["timestamp_iso"],
                    "category": row["category"],
                    "level": row["level"],
                    "message": row["message"],
                    # meta used to be silently dropped from the CSV export
                    "meta": json.dumps(row["meta"], default=str),
                })
        return path


_instance: Optional[SystemEventLogger] = None


def get_event_logger() -> SystemEventLogger:
    """Return the process-wide audit logger, creating it on first use."""
    global _instance
    if _instance is None:
        _instance = SystemEventLogger()
    return _instance


class _LazyEventLogger:
    """
    Proxy that forwards attribute access to the lazily built singleton, so
    ``event_logger.log_event(...)`` works without an import-time side effect.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(get_event_logger(), name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<lazy SystemEventLogger>"


event_logger = _LazyEventLogger()
