"""
Alert state machine.

```
  SAFE ──(score > SAFE_UPPER)──────────────► CAUTION
   ▲                                            │
   │  (score <= SAFE_EXIT)                      │ (score >= THREAT_LOWER sustained)
   └────────────────────────────────────────────┤
                                                ▼
                                             THREAT ──(still >= THREAT_LOWER after
                                                ▲      ALERT_ESCALATE_SECS)──► ALERT ACTIVE
   (score >= THREAT_LOWER, re-escalation)       │                                   │
                                                │                                   │ (score <
                                             RECOVERY ◄────(score < CAUTION_UPPER    │  CAUTION_UPPER
                                                │           for RECOVERY_CONSECUTIVE │  for that long)
                                                │           _SECS)                  │
                                                └──(score <= SAFE_UPPER)──► SAFE ◄──┘
```

Fixes relative to the predecessor implementation
------------------------------------------------
* `_high_since` is cleared whenever the score drops below the THREAT band, so
  persistence is measured on a *contiguous* high period, not from some earlier
  unrelated moment.
* Every transition resets the timer that belongs to the state being entered.
* SAFE → CAUTION happens on the first over-threshold sample (previously it was
  skipped by an `if/elif` chain and only fired on the second sample).
* A **dispatch cooldown** prevents one long incident from re-notifying on every
  oscillation; the transition event carries ``dispatch=True`` only when a
  notification should actually be sent.
* Uses ``time.monotonic()`` so a wall-clock adjustment cannot corrupt timers.

The class is pure logic: it never performs I/O.  The caller decides what to do
with ``event["dispatch"]``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from threatwatch.config import (
    ALERT_COOLDOWN_SECS,
    ALERT_ESCALATE_SECS,
    CAUTION_UPPER_BOUND,
    RECOVERY_CONSECUTIVE_SECS,
    SAFE_UPPER_BOUND,
    THREAT_CONSECUTIVE_SECS,
    THREAT_LOWER_BOUND,
)

STATE_SAFE = "SAFE"
STATE_CAUTION = "CAUTION"
STATE_THREAT = "THREAT"
STATE_ALERT = "ALERT ACTIVE"
STATE_RECOVERY = "RECOVERY"

# Hysteresis: leaving CAUTION needs a slightly lower score than entering it,
# so a score sitting exactly on the boundary cannot flap.
CAUTION_EXIT_BOUND = max(0.0, SAFE_UPPER_BOUND - 0.05)


class AlertStateMachine:
    """Finite state machine turning a continuous score into discrete alert states."""

    STATES = [STATE_SAFE, STATE_CAUTION, STATE_THREAT, STATE_ALERT, STATE_RECOVERY]

    def __init__(self, cooldown_secs: float = ALERT_COOLDOWN_SECS):
        self.cooldown_secs = float(cooldown_secs)
        self.current_state = STATE_SAFE
        self._state_since = time.monotonic()
        self._high_since: Optional[float] = None
        self._low_since: Optional[float] = None
        self._last_dispatch_at: float = -1e9

        self.total_alerts_triggered = 0
        self.emergency_dispatched = False

    # ------------------------------------------------------------------ #
    def update(self, fused_threat_score: float, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Advance the machine.  Returns a transition event dict, or ``None``."""
        now = time.monotonic() if now is None else float(now)
        score = float(fused_threat_score)

        # Persistence timer for the THREAT band, maintained independently of state.
        if score >= THREAT_LOWER_BOUND:
            if self._high_since is None:
                self._high_since = now
        else:
            self._high_since = None

        old_state = self.current_state
        event: Optional[Dict[str, Any]] = None

        if self.current_state == STATE_SAFE:
            if self._high_since is not None and (now - self._high_since) >= THREAT_CONSECUTIVE_SECS:
                event = self._enter(STATE_THREAT, now, "CRITICAL",
                                    f"Score in the THREAT band for {THREAT_CONSECUTIVE_SECS:g}s. "
                                    f"Escalated to THREAT.")
            elif score > SAFE_UPPER_BOUND:
                event = self._enter(STATE_CAUTION, now, "WARNING",
                                    "Elevated distress across modalities. Entered CAUTION.")

        elif self.current_state == STATE_CAUTION:
            if self._high_since is not None and (now - self._high_since) >= THREAT_CONSECUTIVE_SECS:
                event = self._enter(STATE_THREAT, now, "CRITICAL",
                                    f"High distress persisted for {THREAT_CONSECUTIVE_SECS:g}s. "
                                    f"THREAT state entered.")
            elif score <= CAUTION_EXIT_BOUND:
                event = self._enter(STATE_SAFE, now, "INFO",
                                    "Distress returned to baseline. System SAFE.")

        elif self.current_state == STATE_THREAT:
            if score < CAUTION_UPPER_BOUND:
                event = self._enter(STATE_RECOVERY, now, "INFO",
                                    "Distress declining. Entering RECOVERY.")
            elif (now - self._state_since) >= ALERT_ESCALATE_SECS:
                self.total_alerts_triggered += 1
                self.emergency_dispatched = True
                should_dispatch = (now - self._last_dispatch_at) >= self.cooldown_secs
                if should_dispatch:
                    self._last_dispatch_at = now
                event = self._enter(
                    STATE_ALERT, now, "EMERGENCY",
                    "THREAT CONFIRMED: local alarm raised"
                    + (" and emergency alert dispatched." if should_dispatch
                       else " (dispatch suppressed by cooldown)."),
                )
                event["dispatch"] = should_dispatch

        elif self.current_state == STATE_ALERT:
            if score < CAUTION_UPPER_BOUND:
                if self._low_since is None:
                    self._low_since = now
                elif (now - self._low_since) >= RECOVERY_CONSECUTIVE_SECS:
                    self._low_since = None
                    event = self._enter(STATE_RECOVERY, now, "INFO",
                                        f"Distress below threshold for {RECOVERY_CONSECUTIVE_SECS:g}s. "
                                        f"Recovery initialised.")
            else:
                self._low_since = None

        elif self.current_state == STATE_RECOVERY:
            if score <= SAFE_UPPER_BOUND:
                self.emergency_dispatched = False
                self._low_since = None
                event = self._enter(STATE_SAFE, now, "INFO",
                                    "Full recovery verified. System reset to SAFE baseline.")
            elif score >= THREAT_LOWER_BOUND:
                event = self._enter(STATE_THREAT, now, "CRITICAL",
                                    "Re-escalation during recovery. Returning to THREAT.")

        if event is not None:
            event.update({
                "timestamp": now,
                "epoch": time.time(),
                "old_state": old_state,
                "new_state": self.current_state,
                "threat_score": round(score, 3),
            })
            event.setdefault("dispatch", False)

        return event

    # ------------------------------------------------------------------ #
    def _enter(self, new_state: str, now: float, level: str, message: str) -> Dict[str, Any]:
        """Transition and reset every timer that does not survive the transition."""
        self.current_state = new_state
        self._state_since = now
        # The low-score timer only means something while inside ALERT ACTIVE.
        if new_state != STATE_ALERT:
            self._low_since = None
        # Re-entering the THREAT family should not inherit an old high-timer.
        if new_state in (STATE_THREAT, STATE_ALERT, STATE_RECOVERY):
            self._high_since = None
        return {"level": level, "message": message}

    def reset_state(self) -> None:
        """Manual operator reset."""
        self.current_state = STATE_SAFE
        self._state_since = time.monotonic()
        self._high_since = None
        self._low_since = None
        self.emergency_dispatched = False

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "current_state": self.current_state,
            "seconds_in_state": round(time.monotonic() - self._state_since, 1),
            "emergency_dispatched": self.emergency_dispatched,
            "total_alerts": self.total_alerts_triggered,
            "cooldown_secs": self.cooldown_secs,
            "in_threat_band_for": (
                round(time.monotonic() - self._high_since, 2) if self._high_since is not None else None
            ),
        }
