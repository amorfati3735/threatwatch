"""
Alert delivery tests.

A notification system that cannot be tested is a liability during a live demo,
so every behaviour the project relies on is asserted here: cooldown,
de-duplication, failure isolation and the configured/unconfigured fallbacks.
"""

from __future__ import annotations

import time

from threatwatch.alerts.base import AlertMessage, ConsoleNotifier, Notifier
from threatwatch.alerts.manager import AlertDispatcher, build_notifiers
from threatwatch.config import AlertSettings


class _RecordingNotifier(Notifier):
    name = "recording"

    def __init__(self, ok: bool = True, configured: bool = True):
        self.ok = ok
        self._configured = configured
        self.messages = []

    def is_configured(self) -> bool:
        return self._configured

    def send(self, message: AlertMessage) -> bool:
        self.messages.append(message)
        return self.ok


class _ExplodingNotifier(Notifier):
    name = "exploding"

    def send(self, message: AlertMessage) -> bool:
        raise RuntimeError("network is on fire")


def _msg(score: float = 0.9, state: str = "ALERT ACTIVE") -> AlertMessage:
    return AlertMessage(title="Threat detected", body="body", level="EMERGENCY",
                        state=state, threat_score=score)


def test_message_text_contains_key_fields():
    text = _msg().to_text()
    assert "EMERGENCY" in text
    assert "ALERT ACTIVE" in text
    assert "90.0%" in text


def test_dispatch_sends_once():
    n = _RecordingNotifier()
    d = AlertDispatcher([n], cooldown_secs=0.0, background=False, dedup_window_secs=0.0)
    assert d.dispatch_sync(_msg()) is True
    assert len(n.messages) == 1
    assert d.sent_count == 1


def test_dedup_suppresses_repeat_signature():
    n = _RecordingNotifier()
    d = AlertDispatcher([n], cooldown_secs=0.0, dedup_window_secs=60.0, background=False)
    assert d.dispatch_sync(_msg()) is True
    assert d.dispatch_sync(_msg()) is False, "an identical alert must be de-duplicated"
    assert len(n.messages) == 1
    assert d.suppressed_count == 1


def test_cooldown_suppresses_different_alert():
    n = _RecordingNotifier()
    d = AlertDispatcher([n], cooldown_secs=60.0, dedup_window_secs=0.0, background=False)
    assert d.dispatch_sync(_msg()) is True
    assert d.dispatch_sync(_msg(state="THREAT")) is False, "cooldown must gate any alert"
    assert len(n.messages) == 1


def test_cooldown_expires():
    n = _RecordingNotifier()
    d = AlertDispatcher([n], cooldown_secs=0.05, dedup_window_secs=0.0, background=False)
    assert d.dispatch_sync(_msg()) is True
    time.sleep(0.06)
    assert d.dispatch_sync(_msg(state="THREAT")) is True
    assert len(n.messages) == 2


def test_unconfigured_notifier_is_skipped():
    n = _RecordingNotifier(configured=False)
    d = AlertDispatcher([n], cooldown_secs=0.0, background=False)
    assert d.dispatch_sync(_msg()) is False
    assert n.messages == []
    assert d.history[-1]["status"] == "suppressed"


def test_backend_failure_is_isolated():
    """One broken backend must not prevent the others from delivering."""
    bad = _ExplodingNotifier()
    good = _RecordingNotifier()
    d = AlertDispatcher([bad, good], cooldown_secs=0.0, background=False)
    assert d.dispatch_sync(_msg()) is True
    assert len(good.messages) == 1


def test_all_backends_failing_is_recorded_not_raised():
    d = AlertDispatcher([_RecordingNotifier(ok=False)], cooldown_secs=0.0, background=False)
    assert d.dispatch_sync(_msg()) is False
    assert d.failed_count == 1
    assert d.history[-1]["status"] == "failed"


def test_audit_callback_receives_entries():
    seen = []
    d = AlertDispatcher([_RecordingNotifier()], cooldown_secs=0.0, background=False,
                        audit=seen.append)
    d.dispatch_sync(_msg())
    assert seen and seen[0]["status"] == "sent"


def test_build_message_from_state_event():
    d = AlertDispatcher([_RecordingNotifier()], background=False)
    event = {"level": "EMERGENCY", "message": "persisted", "old_state": "THREAT",
             "new_state": "ALERT ACTIVE", "threat_score": 0.87}
    m = d.build_message(event, source="wesad_replay")
    assert m.state == "ALERT ACTIVE"
    assert abs(m.threat_score - 0.87) < 1e-9
    assert "THREAT -> ALERT ACTIVE" in m.body


def test_disabled_alerts_use_null_notifier():
    class Null(Notifier):
        name = "null"

        def send(self, message):
            return False

    notifiers = build_notifiers(AlertSettings(enabled=False))
    assert len(notifiers) == 1
    assert notifiers[0].is_configured()  # NullNotifier configures fine and no-ops


def test_telegram_without_token_falls_back_to_console():
    cfg = AlertSettings(enabled=True, channel="telegram", telegram_bot_token="", telegram_chat_id="")
    notifiers = build_notifiers(cfg)
    assert len(notifiers) == 1
    assert isinstance(notifiers[0], ConsoleNotifier)


def test_telegram_with_credentials_is_selected():
    cfg = AlertSettings(enabled=True, channel="telegram",
                        telegram_bot_token="123:abc", telegram_chat_id="42")
    notifiers = build_notifiers(cfg)
    assert notifiers[0].name == "telegram"
    assert notifiers[0].is_configured()


def test_unknown_channel_falls_back_to_console():
    cfg = AlertSettings(enabled=True, channel="carrier-pigeon")
    notifiers = build_notifiers(cfg)
    assert notifiers[0].name == "console"
