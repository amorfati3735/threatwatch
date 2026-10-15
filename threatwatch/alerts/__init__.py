"""Pluggable alert delivery: message model, notifiers and the dispatch manager."""

from threatwatch.alerts.base import (
    AlertMessage,
    ConsoleNotifier,
    Notifier,
    NullNotifier,
)
from threatwatch.alerts.manager import (
    AlertDispatcher,
    build_dispatcher,
    build_notifiers,
)
from threatwatch.alerts.ntfy import NtfyNotifier
from threatwatch.alerts.telegram import TelegramNotifier

__all__ = [
    "AlertMessage",
    "AlertDispatcher",
    "ConsoleNotifier",
    "Notifier",
    "NullNotifier",
    "NtfyNotifier",
    "TelegramNotifier",
    "build_dispatcher",
    "build_notifiers",
]
