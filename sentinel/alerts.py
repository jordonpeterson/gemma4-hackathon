"""Alert delivery: DB row + macOS notification. Email is a stub."""
import logging
import subprocess
import sys

from sentinel import db

log = logging.getLogger(__name__)


def _notify_macos(title: str, message: str) -> None:
    if sys.platform != "darwin":
        log.info("non-macOS platform, skipping notification: %s", message)
        return
    # osascript treats the strings as AppleScript literals; escape quotes/backslashes.
    esc_msg = message.replace("\\", "\\\\").replace('"', '\\"')
    esc_title = title.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{esc_msg}" with title "{esc_title}"'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=10,
                       capture_output=True)
    except Exception:
        log.exception("osascript notification failed")


def fire(rule_id: int, evaluation_id: int, message: str) -> int:
    """Insert alert row and fire a local notification. Returns alert id."""
    alert_id = db.create_alert(rule_id, evaluation_id, message)
    _notify_macos("Sentinel alert", message)
    log.warning("ALERT #%s (rule %s): %s", alert_id, rule_id, message)
    return alert_id


def send_email(to: str, subject: str, body: str) -> None:
    # TODO: wire up SMTP (or a local mail relay) post-MVP. Deliberately a stub —
    # the MVP alert channels are macOS notifications + the alerts table.
    raise NotImplementedError("email alerts are a post-MVP TODO")
