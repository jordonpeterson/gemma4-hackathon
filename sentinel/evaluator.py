"""Rule evaluation — the important split:

- numeric / boolean rules: pure Python, NEVER calls the LLM
  (model_answer stays NULL in the evaluations row — that's asserted in tests
  and in the definition of done).
- image rules: one llm.ask_image call; "unsure" writes an 'error' row and
  never alerts.

Every evaluation writes an evaluations row, triggered or not. active_hours
and cooldown_minutes gate alert *creation*, not evaluation.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sentinel import alerts, db, llm

log = logging.getLogger(__name__)


def _now_local() -> datetime:
    return datetime.now()


def _within_active_hours(rule: dict, now: Optional[datetime] = None) -> bool:
    hours = rule.get("active_hours") or {}
    start = hours.get("start", "00:00")
    end = hours.get("end", "23:59")
    now = now or _now_local()
    cur = now.strftime("%H:%M")
    if start <= end:
        return start <= cur <= end
    # Overnight window, e.g. 22:00–06:00
    return cur >= start or cur <= end


def _parse_db_ts(ts: str) -> datetime:
    """SQLite datetime('now') is UTC, format 'YYYY-MM-DD HH:MM:SS'."""
    return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)


def _cooldown_active(rule_row_id: int, rule: dict,
                     now: Optional[datetime] = None) -> bool:
    cooldown_min = rule.get("cooldown_minutes", 0) or 0
    if cooldown_min <= 0:
        return False
    last = db.last_alert_for_rule(rule_row_id)
    if last is None:
        return False
    now = now or datetime.now(timezone.utc)
    elapsed_min = (now - _parse_db_ts(last["created_at"])).total_seconds() / 60
    return elapsed_min < cooldown_min


def _numeric_triggered(rule: dict, reading: dict, previous: Optional[dict]) -> bool:
    cond = rule["condition"]
    value = reading["value"]
    if value is None:
        raise ValueError("numeric/boolean reading has no value")
    if cond["type"] == "threshold":
        op, target = cond["operator"], cond["value"]
        if op == "lt":
            return value < target
        if op == "gt":
            return value > target
        if op == "eq":
            return value == target
        raise ValueError(f"unknown operator {op!r}")
    if cond["type"] == "state_change":
        to = float(cond["to"]) if cond["to"] is not None else None
        frm = cond.get("from")
        if float(value) != to:
            return False
        if frm is None:
            # No 'from' constraint: any transition INTO 'to' counts, but it
            # must be a transition — a first-ever reading or a repeat doesn't.
            return previous is not None and float(previous["value"]) != to
        return previous is not None and float(previous["value"]) == float(frm)
    raise ValueError(f"condition type {cond['type']!r} invalid for numeric/boolean")


def evaluate(rule_row: dict, reading: dict) -> dict:
    """Evaluate one rule against one reading.

    Returns {"evaluation_id", "result", "alerted", "alert_id"}.
    """
    rule = json.loads(rule_row["parsed_json"])
    result = "error"
    model_answer = None
    latency_ms = None
    alerted = False
    alert_id = None

    try:
        if rule["modality"] in ("numeric", "boolean"):
            previous = db.previous_reading(reading["sensor_id"], reading["id"])
            result = "triggered" if _numeric_triggered(rule, reading, previous) else "ok"
        elif rule["modality"] == "image":
            if not reading.get("image_path"):
                raise ValueError("image rule evaluated against a reading with no image")
            answer = llm.ask_image(reading["image_path"], rule["condition"]["question"])
            model_answer = json.dumps({"answer": answer["answer"], "reason": answer["reason"]})
            latency_ms = answer.get("latency_ms")
            if answer["answer"] == "yes":
                result = "triggered"
            elif answer["answer"] == "no":
                result = "ok"
            else:  # unsure -> error row for review, never an alert
                result = "error"
        else:
            raise ValueError(f"unknown modality {rule['modality']!r}")
    except Exception as exc:
        log.exception("evaluation failed for rule %s", rule_row["id"])
        result = "error"
        if model_answer is None:
            model_answer = json.dumps({"error": str(exc)}) if rule["modality"] == "image" else None

    evaluation_id = db.create_evaluation(
        rule_row["id"], reading["id"], result,
        model_answer=model_answer, latency_ms=latency_ms,
    )

    if result == "triggered":
        if not _within_active_hours(rule):
            log.info("rule %s triggered outside active hours — no alert", rule_row["id"])
        elif _cooldown_active(rule_row["id"], rule):
            log.info("rule %s in cooldown — no alert", rule_row["id"])
        else:
            alert_id = alerts.fire(rule_row["id"], evaluation_id,
                                   rule["action"]["message"])
            alerted = True

    return {"evaluation_id": evaluation_id, "result": result,
            "alerted": alerted, "alert_id": alert_id}
