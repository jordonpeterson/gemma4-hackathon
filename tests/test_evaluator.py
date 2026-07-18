"""Evaluator tests: pure-Python numeric path, LLM image path (mocked),
cooldown and active-hours suppression. No model required."""
import json
from datetime import datetime

from sentinel import db, evaluator, llm


def _mk_rule(sensor_id, rule_dict, status="active"):
    rid = db.create_rule(sensor_id, "test instruction", json.dumps(rule_dict), status=status)
    return db.get_rule(rid)


def _threshold_rule(sensor="keg_scale", op="lt", value=15, cooldown=240, hours=None):
    r = {
        "sensor": sensor, "modality": "numeric",
        "condition": {"type": "threshold", "operator": op, "value": value},
        "action": {"type": "alert", "message": "Keg low"},
        "cooldown_minutes": cooldown,
    }
    if hours:
        r["active_hours"] = hours
    return r


# ---------- threshold ----------

def test_threshold_triggers_and_alerts_without_llm(demo_sensors):
    rule = _mk_rule(demo_sensors["keg_scale"], _threshold_rule())
    rd = db.get_reading(db.create_reading(demo_sensors["keg_scale"], "numeric", value=12))
    out = evaluator.evaluate(rule, rd)
    assert out["result"] == "triggered"
    assert out["alerted"] is True
    ev = db.last_evaluation_for_rule(rule["id"])
    assert ev["result"] == "triggered"
    assert ev["model_answer"] is None  # definition of done #4: no LLM involved
    assert db.last_alert_for_rule(rule["id"])["message"] == "Keg low"


def test_threshold_no_trigger(demo_sensors):
    rule = _mk_rule(demo_sensors["keg_scale"], _threshold_rule())
    rd = db.get_reading(db.create_reading(demo_sensors["keg_scale"], "numeric", value=40))
    out = evaluator.evaluate(rule, rd)
    assert out["result"] == "ok"
    assert out["alerted"] is False
    assert db.last_alert_for_rule(rule["id"]) is None


def test_threshold_gt_and_eq(demo_sensors):
    sid = demo_sensors["keg_scale"]
    gt = _mk_rule(sid, _threshold_rule(op="gt", value=90))
    rd = db.get_reading(db.create_reading(sid, "numeric", value=95))
    assert evaluator.evaluate(gt, rd)["result"] == "triggered"
    eq = _mk_rule(sid, _threshold_rule(op="eq", value=95))
    rd2 = db.get_reading(db.create_reading(sid, "numeric", value=95))
    assert evaluator.evaluate(eq, rd2)["result"] == "triggered"


# ---------- state change ----------

def _state_rule(frm, to):
    return {
        "sensor": "door_state", "modality": "boolean",
        "condition": {"type": "state_change", "from": frm, "to": to},
        "action": {"type": "alert", "message": "Door state changed"},
        "cooldown_minutes": 0,
    }


def test_state_change_triggers_on_transition(demo_sensors):
    sid = demo_sensors["door_state"]
    rule = _mk_rule(sid, _state_rule(True, False))
    db.create_reading(sid, "boolean", value=1)
    rd = db.get_reading(db.create_reading(sid, "boolean", value=0))
    assert evaluator.evaluate(rule, rd)["result"] == "triggered"


def test_state_change_needs_matching_previous(demo_sensors):
    sid = demo_sensors["door_state"]
    rule = _mk_rule(sid, _state_rule(True, False))
    # No previous reading at all -> not a transition.
    rd = db.get_reading(db.create_reading(sid, "boolean", value=0))
    assert evaluator.evaluate(rule, rd)["result"] == "ok"
    # Previous is already 0 -> no transition either.
    rd2 = db.get_reading(db.create_reading(sid, "boolean", value=0))
    assert evaluator.evaluate(rule, rd2)["result"] == "ok"


def test_state_change_without_from(demo_sensors):
    sid = demo_sensors["door_state"]
    rule = _mk_rule(sid, {
        "sensor": "door_state", "modality": "boolean",
        "condition": {"type": "state_change", "to": True},
        "action": {"type": "alert", "message": "Door on"},
        "cooldown_minutes": 0,
    })
    db.create_reading(sid, "boolean", value=0)
    rd = db.get_reading(db.create_reading(sid, "boolean", value=1))
    assert evaluator.evaluate(rule, rd)["result"] == "triggered"
    # Staying at 1 is not a new transition.
    rd2 = db.get_reading(db.create_reading(sid, "boolean", value=1))
    assert evaluator.evaluate(rule, rd2)["result"] == "ok"


# ---------- cooldown suppression ----------

def test_cooldown_suppresses_realert(demo_sensors):
    sid = demo_sensors["keg_scale"]
    rule = _mk_rule(sid, _threshold_rule(cooldown=240))
    rd1 = db.get_reading(db.create_reading(sid, "numeric", value=10))
    out1 = evaluator.evaluate(rule, rd1)
    assert out1["alerted"] is True
    # Still low next cycle: evaluation happens, alert suppressed.
    rd2 = db.get_reading(db.create_reading(sid, "numeric", value=9))
    out2 = evaluator.evaluate(rule, rd2)
    assert out2["result"] == "triggered"
    assert out2["alerted"] is False
    assert len([a for a in db.list_alerts() if a["rule_id"] == rule["id"]]) == 1


def test_zero_cooldown_realerts(demo_sensors):
    sid = demo_sensors["keg_scale"]
    rule = _mk_rule(sid, _threshold_rule(cooldown=0))
    for v in (10, 9):
        rd = db.get_reading(db.create_reading(sid, "numeric", value=v))
        assert evaluator.evaluate(rule, rd)["alerted"] is True


# ---------- active hours suppression ----------

def test_active_hours_suppress_alert(demo_sensors, monkeypatch):
    sid = demo_sensors["keg_scale"]
    rule = _mk_rule(sid, _threshold_rule(hours={"start": "09:00", "end": "17:00"}))
    monkeypatch.setattr(evaluator, "_now_local",
                        lambda: datetime(2026, 7, 18, 3, 0))  # 03:00, outside window
    rd = db.get_reading(db.create_reading(sid, "numeric", value=10))
    out = evaluator.evaluate(rule, rd)
    assert out["result"] == "triggered"  # evaluation still recorded
    assert out["alerted"] is False


def test_overnight_active_hours_window(demo_sensors, monkeypatch):
    sid = demo_sensors["keg_scale"]
    rule = _mk_rule(sid, _threshold_rule(hours={"start": "22:00", "end": "06:00"}))
    monkeypatch.setattr(evaluator, "_now_local",
                        lambda: datetime(2026, 7, 18, 23, 30))  # inside overnight window
    rd = db.get_reading(db.create_reading(sid, "numeric", value=10))
    assert evaluator.evaluate(rule, rd)["alerted"] is True


# ---------- image path (mocked LLM) ----------

def _image_rule():
    return {
        "sensor": "breakroom_cam", "modality": "image",
        "condition": {"type": "visual_question", "question": "Is the coke box empty?"},
        "action": {"type": "alert", "message": "Restock cokes"},
        "cooldown_minutes": 0,
    }


def _image_reading(demo_sensors, tmp_path):
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    rid = db.create_reading(demo_sensors["breakroom_cam"], "image", image_path=str(p))
    return db.get_reading(rid)


def test_image_yes_triggers(demo_sensors, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "ask_image", lambda p, q: {
        "answer": "yes", "reason": "box is empty", "latency_ms": 1234})
    rule = _mk_rule(demo_sensors["breakroom_cam"], _image_rule())
    out = evaluator.evaluate(rule, _image_reading(demo_sensors, tmp_path))
    assert out["result"] == "triggered"
    assert out["alerted"] is True
    ev = db.last_evaluation_for_rule(rule["id"])
    assert json.loads(ev["model_answer"])["answer"] == "yes"
    assert ev["latency_ms"] == 1234


def test_image_no_is_ok(demo_sensors, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "ask_image", lambda p, q: {
        "answer": "no", "reason": "cans visible", "latency_ms": 1000})
    rule = _mk_rule(demo_sensors["breakroom_cam"], _image_rule())
    out = evaluator.evaluate(rule, _image_reading(demo_sensors, tmp_path))
    assert out["result"] == "ok"
    assert out["alerted"] is False


def test_image_unsure_writes_error_row_not_alert(demo_sensors, tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "ask_image", lambda p, q: {
        "answer": "unsure", "reason": "image too dark", "latency_ms": 900})
    rule = _mk_rule(demo_sensors["breakroom_cam"], _image_rule())
    out = evaluator.evaluate(rule, _image_reading(demo_sensors, tmp_path))
    assert out["result"] == "error"
    assert out["alerted"] is False
    ev = db.last_evaluation_for_rule(rule["id"])
    assert ev["result"] == "error"
    assert db.last_alert_for_rule(rule["id"]) is None
