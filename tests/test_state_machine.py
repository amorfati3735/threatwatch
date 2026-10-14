"""
Alert state machine tests, driven by an explicit simulated clock so the timing
behaviour is deterministic and fast.
"""

from __future__ import annotations

from threatwatch.fusion.state_machine import (
    STATE_ALERT,
    STATE_CAUTION,
    STATE_RECOVERY,
    STATE_SAFE,
    STATE_THREAT,
    AlertStateMachine,
)


def _run(sm: AlertStateMachine, score: float, seconds: float, t: float, dt: float = 0.5):
    events = []
    for _ in range(int(round(seconds / dt))):
        t += dt
        e = sm.update(score, now=t)
        if e is not None:
            events.append(e)
    return t, events


def test_starts_safe():
    sm = AlertStateMachine()
    assert sm.current_state == STATE_SAFE
    assert sm.get_state_summary()["total_alerts"] == 0


def test_first_over_threshold_sample_enters_caution():
    """
    Regression: the predecessor's if/elif chain skipped CAUTION on the first
    over-threshold sample and only fired on the second.
    """
    sm = AlertStateMachine()
    e = sm.update(0.50, now=0.5)
    assert e is not None
    assert e["new_state"] == STATE_CAUTION
    assert e["old_state"] == STATE_SAFE


def test_caution_returns_to_safe_with_hysteresis():
    sm = AlertStateMachine()
    sm.update(0.50, now=0.5)                       # -> CAUTION
    t, _ = _run(sm, 0.36, 1.0, 0.5)                # inside the hysteresis band
    assert sm.current_state == STATE_CAUTION, "must not flap back on a boundary score"
    t, ev = _run(sm, 0.20, 1.0, t)                 # clearly low -> SAFE
    assert sm.current_state == STATE_SAFE


def test_full_cycle_and_single_dispatch():
    sm = AlertStateMachine(cooldown_secs=30.0)
    t = 0.0

    t, ev = _run(sm, 0.50, 0.5, t)
    assert sm.current_state == STATE_CAUTION

    # Hold inside the THREAT band long enough to satisfy persistence.
    t, ev = _run(sm, 0.80, 3.0, t)
    assert any(e["new_state"] == STATE_THREAT for e in ev), "persistence should escalate"
    assert sm.current_state == STATE_THREAT

    # Keep holding: THREAT -> ALERT ACTIVE, and this is the one dispatch.
    t, ev = _run(sm, 0.80, 2.0, t)
    alerts = [e for e in ev if e["new_state"] == STATE_ALERT]
    assert alerts, "should reach ALERT ACTIVE"
    assert alerts[0]["dispatch"] is True
    assert alerts[0]["level"] == "EMERGENCY"
    assert sm.total_alerts_triggered == 1

    # Still alarming: the cooldown must prevent a second notification.
    t, ev = _run(sm, 0.90, 5.0, t)
    assert not any(e.get("dispatch") for e in ev), "cooldown must suppress re-notification"

    # Drop -> ALERT ACTIVE requires the recovery window before RECOVERY.
    t, ev = _run(sm, 0.50, 4.0, t)
    assert sm.current_state == STATE_RECOVERY

    # Fully low -> SAFE and flags cleared.
    t, ev = _run(sm, 0.20, 1.0, t)
    assert sm.current_state == STATE_SAFE
    assert sm.emergency_dispatched is False


def test_transient_spike_never_escalates():
    """A single high sample must not become a THREAT."""
    sm = AlertStateMachine()
    t = 0.0
    t, _ = _run(sm, 0.95, 0.5, t)      # one tick high -> CAUTION
    assert sm.current_state == STATE_CAUTION
    t, ev = _run(sm, 0.10, 6.0, t)     # back to normal
    assert sm.current_state == STATE_SAFE
    assert sm.total_alerts_triggered == 0


def test_re_escalation_from_recovery():
    sm = AlertStateMachine()
    t = 0.0
    t, _ = _run(sm, 0.90, 4.0, t)      # CAUTION -> THREAT
    t, _ = _run(sm, 0.90, 2.0, t)      # -> ALERT ACTIVE
    t, _ = _run(sm, 0.40, 4.0, t)      # -> RECOVERY
    assert sm.current_state == STATE_RECOVERY
    t, ev = _run(sm, 0.90, 0.5, t)     # spikes again
    assert sm.current_state == STATE_THREAT, "a spike during recovery must re-escalate"


def test_manual_reset():
    sm = AlertStateMachine()
    _run(sm, 0.90, 6.0, 0.0)
    assert sm.current_state in (STATE_THREAT, STATE_ALERT)
    sm.reset_state()
    assert sm.current_state == STATE_SAFE
    assert sm.emergency_dispatched is False


def test_cooldown_expiry_allows_second_dispatch():
    sm = AlertStateMachine(cooldown_secs=5.0)
    t = 0.0
    t, _ = _run(sm, 0.90, 4.0, t)      # -> THREAT
    t, ev = _run(sm, 0.90, 2.0, t)     # -> ALERT ACTIVE (dispatch 1)
    assert sum(1 for e in ev if e.get("dispatch")) == 1

    t, _ = _run(sm, 0.40, 4.0, t)      # -> RECOVERY
    t, _ = _run(sm, 0.20, 1.0, t)      # -> SAFE
    t, _ = _run(sm, 0.90, 4.0, t)      # -> THREAT again
    t, ev = _run(sm, 0.90, 2.0, t)     # -> ALERT ACTIVE (dispatch 2, cooldown elapsed)
    assert sum(1 for e in ev if e.get("dispatch")) == 1
    assert sm.total_alerts_triggered == 2
