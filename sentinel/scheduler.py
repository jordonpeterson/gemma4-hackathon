"""Background loop: ingest inbox images -> evaluate active rules with new
readings -> alert. One daemon thread, no Celery, no Redis.
"""
import json
import logging
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sentinel import config, db, evaluator

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

_last_cycle: Optional[str] = None
_stop = threading.Event()
_thread: Optional[threading.Thread] = None


def last_cycle_time() -> Optional[str]:
    return _last_cycle


def ingest_inbox() -> int:
    """Pick up new files from inbox/<sensor_name>/, move them into
    data/images/, and create readings rows. Returns number ingested."""
    inbox = Path(config.INBOX_DIR)
    images_dir = Path(config.IMAGES_DIR)
    images_dir.mkdir(parents=True, exist_ok=True)
    if not inbox.exists():
        return 0
    count = 0
    for sensor_dir in sorted(p for p in inbox.iterdir() if p.is_dir()):
        sensor = db.get_sensor_by_name(sensor_dir.name)
        if sensor is None or sensor["kind"] != "image":
            log.warning("inbox dir %s does not match an image sensor — skipping",
                        sensor_dir.name)
            continue
        for f in sorted(sensor_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            dest = images_dir / f"{sensor_dir.name}_{ts}{f.suffix.lower()}"
            try:
                shutil.move(str(f), str(dest))
            except OSError:
                log.exception("could not move %s", f)
                continue
            db.create_reading(sensor["id"], "image", image_path=str(dest))
            count += 1
            log.info("ingested %s for sensor %s", dest.name, sensor_dir.name)
    return count


def evaluate_due_rules() -> int:
    """Evaluate every active rule whose sensor has a reading newer than the
    rule's last evaluation. Returns number of evaluations run."""
    ran = 0
    for rule_row in db.list_rules(status="active"):
        try:
            rule = json.loads(rule_row["parsed_json"])
        except json.JSONDecodeError:
            log.error("rule %s has corrupt parsed_json — skipping", rule_row["id"])
            continue
        sensor = db.get_sensor_by_name(rule["sensor"])
        if sensor is None:
            continue
        reading = db.latest_reading(sensor["id"])
        if reading is None:
            continue
        last_eval = db.last_evaluation_for_rule(rule_row["id"])
        if last_eval is not None and last_eval["reading_id"] >= reading["id"]:
            continue  # nothing new since we last looked
        outcome = evaluator.evaluate(rule_row, reading)
        ran += 1
        log.info("rule %s -> %s%s", rule_row["id"], outcome["result"],
                 " (ALERT)" if outcome["alerted"] else "")
    return ran


def run_cycle() -> dict:
    global _last_cycle
    ingested = ingest_inbox()
    evaluated = evaluate_due_rules()
    _last_cycle = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {"ingested": ingested, "evaluated": evaluated, "at": _last_cycle}


def _loop() -> None:
    log.info("scheduler started (poll every %ss)", config.POLL_SECONDS)
    while not _stop.is_set():
        try:
            run_cycle()
        except Exception:
            log.exception("scheduler cycle failed")
        _stop.wait(config.POLL_SECONDS)


def start() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="sentinel-scheduler", daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=5)
