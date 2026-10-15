"""Unit tests for NtfyNotifier."""

from unittest.mock import MagicMock, patch

from threatwatch.alerts.base import AlertMessage
from threatwatch.alerts.ntfy import NtfyNotifier


def test_ntfy_not_configured():
    n = NtfyNotifier(topic="")
    assert not n.is_configured()
    msg = AlertMessage(title="Test", body="Body")
    assert not n.send(msg)
    assert "missing NTFY_TOPIC" in n.last_error


def test_ntfy_delivery_success():
    n = NtfyNotifier(topic="threatwatch_alerts", server="https://ntfy.sh")
    assert n.is_configured()

    msg = AlertMessage(
        title="Emergency Escalation",
        body="High distress and agitation detected.",
        level="EMERGENCY",
        threat_score=0.92,
    )

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        ok = n.send(msg)
        assert ok is True
        assert n.sent == 1
        assert mock_post.called

        args, kwargs = mock_post.call_args
        assert args[0] == "https://ntfy.sh/threatwatch_alerts"
        assert kwargs["headers"]["Priority"] == "urgent"
        assert "warning" in kwargs["headers"]["Tags"]
