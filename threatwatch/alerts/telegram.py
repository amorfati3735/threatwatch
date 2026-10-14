"""
Telegram notifier.

Uses the Telegram Bot API directly (no SDK).  Telegram is the default channel
because it is free, needs no paid account or trial credit, and works from WSL
with a single HTTPS POST.

Setup (one minute):
  1. Message ``@BotFather`` on Telegram, send ``/newbot``, copy the token.
  2. Send any message to your new bot, then open
     ``https://api.telegram.org/bot<TOKEN>/getUpdates`` and copy
     ``result[0].message.chat.id``.
  3. Put both into ``.env``:
        ALERTS_ENABLED=1
        ALERT_CHANNEL=telegram
        TELEGRAM_BOT_TOKEN=123456:ABC...
        TELEGRAM_CHAT_ID=987654321

A delivery failure is logged and swallowed: a dead network must never take the
detection pipeline down with it.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from threatwatch.alerts.base import AlertMessage, Notifier

API_ROOT = "https://api.telegram.org"


class TelegramNotifier(Notifier):
    """Delivers alerts to a Telegram chat via the Bot API."""

    name = "telegram"

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout_secs: float = 8.0,
        max_retries: int = 2,
    ):
        self.bot_token = (bot_token or "").strip()
        self.chat_id = str(chat_id or "").strip()
        self.timeout_secs = float(timeout_secs)
        self.max_retries = int(max_retries)
        self.last_error = ""
        self.sent = 0

    # ------------------------------------------------------------------ #
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d["chat_id"] = self.chat_id or None
        d["last_error"] = self.last_error or None
        d["sent"] = self.sent
        return d

    # ------------------------------------------------------------------ #
    def _post(self, method: str, payload: Dict[str, Any], files: Optional[Dict] = None) -> bool:
        try:
            import requests
        except ImportError:
            self.last_error = "requests is not installed"
            return False

        url = f"{API_ROOT}/bot{self.bot_token}/{method}"
        for attempt in range(self.max_retries + 1):
            try:
                if files:
                    resp = requests.post(url, data=payload, files=files, timeout=self.timeout_secs)
                else:
                    resp = requests.post(url, json=payload, timeout=self.timeout_secs)
                if resp.status_code == 200 and resp.json().get("ok", False):
                    return True
                self.last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as exc:
                self.last_error = str(exc)
            if attempt < self.max_retries:
                time.sleep(0.5 * (2 ** attempt))
        print(f"[telegram] delivery failed: {self.last_error}")
        return False

    # ------------------------------------------------------------------ #
    def send(self, message: AlertMessage) -> bool:
        if not self.is_configured():
            self.last_error = "missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
            print(f"[telegram] not configured: {self.last_error}")
            return False

        text = message.to_text()

        # A photo with a caption is far more useful than text alone when the
        # alert came from the vision path.
        if message.image_path is not None:
            try:
                from pathlib import Path

                p = Path(message.image_path)
                if p.exists():
                    ok = self._post(
                        "sendPhoto",
                        {"chat_id": self.chat_id, "caption": text[:1024]},
                        files={"photo": (p.name, p.open("rb"), "image/jpeg")},
                    )
                    if ok:
                        self.sent += 1
                        return True
            except Exception as exc:
                self.last_error = f"photo path failed, falling back to text: {exc}"

        ok = self._post("sendMessage", {"chat_id": self.chat_id, "text": text})
        if ok:
            self.sent += 1
        return ok
