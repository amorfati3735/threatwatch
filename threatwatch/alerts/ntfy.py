"""
ntfy.sh notifier.

Delivers alerts to an ntfy topic via HTTP POST.
Triggers Tasker, webhooks, or push notifications to mobile devices.

Setup:
  Put into `.env`:
    ALERTS_ENABLED=1
    ALERT_CHANNEL=ntfy  (or ALERT_CHANNEL=telegram,ntfy)
    NTFY_TOPIC=your_topic_name
    NTFY_SERVER=https://ntfy.sh       # optional, default https://ntfy.sh
    NTFY_TOKEN=                       # optional, if topic requires bearer auth
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from threatwatch.alerts.base import AlertMessage, Notifier


class NtfyNotifier(Notifier):
    """Delivers alerts to an ntfy topic."""

    name = "ntfy"

    def __init__(
        self,
        topic: str,
        server: str = "https://ntfy.sh",
        token: Optional[str] = None,
        timeout_secs: float = 8.0,
        max_retries: int = 2,
    ):
        self.topic = (topic or "").strip().lstrip("/")
        self.server = (server or "https://ntfy.sh").rstrip("/")
        self.token = (token or "").strip()
        self.timeout_secs = float(timeout_secs)
        self.max_retries = int(max_retries)
        self.last_error = ""
        self.sent = 0

    def is_configured(self) -> bool:
        return bool(self.topic)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["topic"] = self.topic or None
        d["server"] = self.server
        d["last_error"] = self.last_error or None
        d["sent"] = self.sent
        return d

    def _post(self, text: str, headers: Dict[str, str], files: Optional[Dict] = None) -> bool:
        try:
            import requests
        except ImportError:
            self.last_error = "requests is not installed"
            return False

        url = f"{self.server}/{self.topic}"
        auth_headers = dict(headers)
        if self.token:
            auth_headers["Authorization"] = f"Bearer {self.token}"

        for attempt in range(self.max_retries + 1):
            try:
                if files:
                    resp = requests.post(url, data=text.encode("utf-8"), headers=auth_headers, files=files, timeout=self.timeout_secs)
                else:
                    resp = requests.post(url, data=text.encode("utf-8"), headers=auth_headers, timeout=self.timeout_secs)
                if resp.status_code in (200, 201):
                    return True
                self.last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as exc:
                self.last_error = str(exc)
            if attempt < self.max_retries:
                time.sleep(0.5 * (2 ** attempt))

        print(f"[ntfy] delivery failed: {self.last_error}")
        return False

    def send(self, message: AlertMessage) -> bool:
        if not self.is_configured():
            self.last_error = "missing NTFY_TOPIC"
            print(f"[ntfy] not configured: {self.last_error}")
            return False

        level = (message.level or "").upper()
        if level in ("EMERGENCY", "CRITICAL"):
            priority = "urgent"  # 5 in Tasker / ntfy
            tags = "warning,rotating_light,sos"
        elif level == "WARNING":
            priority = "high"    # 4
            tags = "warning"
        else:
            priority = "default" # 3
            tags = "information_source"

        score_pct = f"{round(message.threat_score * 100)}%" if message.threat_score else "--%"

        headers = {
            "Title": f"ThreatWatch: {message.title} ({score_pct} - {message.level})",
            "Priority": priority,
            "Tags": tags,
            "Click": "http://localhost:8000/",
            "X-Threat-Score": str(round(message.threat_score, 3)),
            "X-Threat-State": str(message.state),
            "X-Threat-Level": str(message.level),
            "X-Sensor-Source": str(message.source),
            "X-Timestamp": str(message.when),
        }

        # Format clean, authoritative text optimized for Tasker SMS emergency contact routing
        emergency_lines = [
            "EMERGENCY DISTRESS ALERT - ThreatWatch",
            f"Status: {message.state} ({message.level})",
            f"Threat Score: {score_pct}",
            f"Trigger: {message.title}",
            f"Sensor Source: {message.source}",
            f"Detected At: {message.when}",
            "",
            message.body.strip(),
            "",
            "Action: Automated distress signal triggered. Check on user immediately.",
        ]
        text = "\n".join(emergency_lines)

        if message.image_path is not None:
            try:
                from pathlib import Path
                p = Path(message.image_path)
                if p.exists():
                    headers["Filename"] = p.name
            except Exception:
                pass

        ok = self._post(text, headers)
        if ok:
            self.sent += 1
        return ok
