"""
Alert dispatch manager.

Sits between the state machine and the notifiers and adds the operational
behaviour a real alerting system needs:

* **cooldown** — one incident must not produce a message per oscillation;
* **de-duplication** — the same (level, state) signature inside a short window is
  suppressed even if the cooldown has elapsed;
* **non-blocking delivery** — a slow or dead network must not stall the 20 FPS
  telemetry loop, so delivery happens on a small worker pool;
* **audit trail** — every attempt (sent / suppressed / failed) is recorded, so
  the write-up and the dashboard can show what actually happened.

The manager never raises into the caller.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from threatwatch.alerts.base import AlertMessage, ConsoleNotifier, Notifier, NullNotifier
from threatwatch.alerts.telegram import TelegramNotifier
from threatwatch.config import AlertSettings, settings as load_settings

AuditFn = Callable[[Dict[str, Any]], None]


class AlertDispatcher:
    """Rate-limited, de-duplicated, non-blocking alert delivery."""

    def __init__(
        self,
        notifiers: Sequence[Notifier],
        cooldown_secs: float = 30.0,
        dedup_window_secs: float = 60.0,
        background: bool = True,
        audit: Optional[AuditFn] = None,
        max_workers: int = 2,
    ):
        self.notifiers: List[Notifier] = [n for n in notifiers]
        self.cooldown_secs = float(cooldown_secs)
        self.dedup_window_secs = float(dedup_window_secs)
        self.background = bool(background)
        self.audit = audit

        self._last_sent_at: float = -1e9
        self._last_signature: str = ""
        self._last_signature_at: float = -1e9
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="alert") if background else None

        self.history: List[Dict[str, Any]] = []
        self.sent_count = 0
        self.suppressed_count = 0
        self.failed_count = 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def signature(message: AlertMessage) -> str:
        raw = f"{message.level}|{message.state}|{message.title}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _suppression_reason(self, message: AlertMessage, now: float) -> Optional[str]:
        if not any(n.is_configured() for n in self.notifiers):
            return "no notifier configured"
        if (now - self._last_sent_at) < self.cooldown_secs:
            return f"cooldown ({self.cooldown_secs:g}s)"
        sig = self.signature(message)
        if sig == self._last_signature and (now - self._last_signature_at) < self.dedup_window_secs:
            return f"duplicate within {self.dedup_window_secs:g}s"
        return None

    def _record(self, message: AlertMessage, status: str, detail: str = "") -> None:
        entry = {
            "timestamp": message.timestamp,
            "level": message.level,
            "state": message.state,
            "title": message.title,
            "threat_score": message.threat_score,
            "status": status,
            "detail": detail,
        }
        self.history.append(entry)
        if len(self.history) > 200:
            self.history.pop(0)
        if self.audit is not None:
            try:
                self.audit(entry)
            except Exception:
                pass
        print(f"[alerts] {status.upper():9s} {message.level:9s} {message.title} {detail}".rstrip())

    # ------------------------------------------------------------------ #
    def dispatch_sync(self, message: AlertMessage) -> bool:
        """Deliver now, on the calling thread.  Used by tests and CLI paths."""
        now = time.time()
        reason = self._suppression_reason(message, now)
        if reason is not None:
            self.suppressed_count += 1
            self._record(message, "suppressed", reason)
            return False

        delivered = False
        errors: List[str] = []
        for notifier in self.notifiers:
            if not notifier.is_configured():
                continue
            try:
                if notifier.send(message):
                    delivered = True
            except Exception as exc:  # a broken backend must not break the others
                errors.append(f"{notifier.name}: {exc}")

        if delivered:
            self.sent_count += 1
            self._last_sent_at = now
            self._last_signature = self.signature(message)
            self._last_signature_at = now
            self._record(message, "sent", ", ".join(n.name for n in self.notifiers))
        else:
            self.failed_count += 1
            self._record(message, "failed", "; ".join(errors) or "no backend delivered")
        return delivered

    def dispatch(self, message: AlertMessage) -> None:
        """Queue delivery.  Never blocks, never raises."""
        try:
            if self._pool is not None:
                self._pool.submit(self.dispatch_sync, message)
            else:
                self.dispatch_sync(message)
        except Exception as exc:
            self._record(message, "failed", f"dispatch error: {exc}")

    # ------------------------------------------------------------------ #
    def build_message(self, event: Dict[str, Any], source: str = "unknown", image_path=None) -> AlertMessage:
        """Turn a state-machine event into a human-readable alert."""
        score = float(event.get("threat_score", 0.0))
        old_state = event.get("old_state", "?")
        new_state = event.get("new_state", "ALERT ACTIVE")
        body = (
            f"A multimodal threat condition has been confirmed.\n\n"
            f"Transition     : {old_state} -> {new_state}\n"
            f"Trigger        : {event.get('message', 'threshold exceeded')}\n\n"
            f"Physiological and visual distress signals agreed above the alarm "
            f"threshold and persisted long enough to rule out a transient spike."
        )
        return AlertMessage(
            title="Threat detected",
            body=body,
            level=str(event.get("level", "EMERGENCY")),
            state=str(new_state),
            threat_score=score,
            source=source,
            image_path=image_path,
            meta={"old_state": old_state, "new_state": new_state},
        )

    def get_state(self) -> Dict[str, Any]:
        return {
            "notifiers": [n.describe() for n in self.notifiers],
            "cooldown_secs": self.cooldown_secs,
            "dedup_window_secs": self.dedup_window_secs,
            "sent": self.sent_count,
            "suppressed": self.suppressed_count,
            "failed": self.failed_count,
            "recent": self.history[-10:],
        }

    def reset_counters(self) -> None:
        self.sent_count = self.suppressed_count = self.failed_count = 0
        self.history.clear()

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_notifiers(cfg: Optional[AlertSettings] = None) -> List[Notifier]:
    """Construct the configured notifier chain, falling back safely."""
    cfg = cfg or load_settings()

    if not cfg.enabled:
        return [NullNotifier()]

    channel = (cfg.channel or "console").strip().lower()

    if channel == "telegram":
        if cfg.telegram_ready:
            return [
                TelegramNotifier(
                    cfg.telegram_bot_token,
                    cfg.telegram_chat_id,
                    timeout_secs=cfg.timeout_secs,
                    max_retries=cfg.max_retries,
                )
            ]
        print(
            "[alerts] ALERT_CHANNEL=telegram but TELEGRAM_BOT_TOKEN / "
            "TELEGRAM_CHAT_ID are unset — falling back to console output."
        )
        return [ConsoleNotifier()]

    if channel == "console":
        return [ConsoleNotifier()]

    print(f"[alerts] unknown ALERT_CHANNEL={channel!r} — using console.")
    return [ConsoleNotifier()]


def build_dispatcher(
    cfg: Optional[AlertSettings] = None,
    audit: Optional[AuditFn] = None,
    background: bool = True,
) -> AlertDispatcher:
    """Build a dispatcher from settings."""
    cfg = cfg or load_settings()
    return AlertDispatcher(
        build_notifiers(cfg),
        cooldown_secs=cfg.cooldown_secs,
        background=background,
        audit=audit,
    )
