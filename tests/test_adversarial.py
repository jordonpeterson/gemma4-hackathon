"""Adversarial tests: scheduler ingest/eval, run_cycle end-to-end, API edges,
evaluator failure paths, strip_fences fuzzing. No real network — LLM is either
monkeypatched or pointed at a dead localhost port.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentinel import config, db, evaluator, llm, scheduler
from sentinel.api import app


# ---------- helpers ----------

def _mk_rule(sensor_id, rule_dict, status="active"):
    rid = db.create_rule(sensor_id, "adversarial", json.dumps(rule_dict), status=status)
    return db.get_rule(rid)


def _threshold(sensor="keg_scale", op="lt", value=15, cooldown=0):
    return {
        "sensor": sensor, "modality": "numeric",
        "condition": {"type": "threshold", "operator": op, "value": value},
        "action": {"type": "alert", "message": "thresh"},
        "cooldown_minutes": cooldown,
    }


def _image_rule(sensor="breakroom_cam", cooldown=0):
    return {
        "sensor": sensor, "modality": "image",
        "condition": {"type": "visual_question", "question": "Is it empty?"},
        "action": {"type": "alert", "message": "img alert"},
        "cooldown_minutes": cooldown,
    }


def _inbox_file(rel, data=b"\x89PNG\r\n\x1a\nfake"):
    p = Path(config.INBOX_DIR) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


# =========================================================================
# 1. scheduler.ingest_inbox
# =========================================================================

def test_ingest_moves_file_and_creates_reading(demo_sensors):
    f = _inbox_file("breakroom_cam/shot.png")
    assert scheduler.ingest_inbox() == 1
    # file physically moved out of inbox
    assert not f.exists()
    rd = db.latest_reading(demo_sensors["breakroom_cam"])
    assert rd is not None and rd["kind"] == "image"
    dest = Path(rd["image_path"])
    assert dest.is_file()
    assert dest.parent == Path(config.IMAGES_DIR)
    assert dest.suffix == ".png"


def test_ingest_uppercase_extension_normalized(demo_sensors):
    _inbox_file("breakroom_cam/SHOT.PNG")
    assert scheduler.ingest_inbox() == 1
    rd = db.latest_reading(demo_sensors["breakroom_cam"])
    assert rd["image_path"].endswith(".png")


def test_ingest_ignores_non_image_extensions(demo_sensors):
    f = _inbox_file("breakroom_cam/notes.txt", b"hello")
    assert scheduler.ingest_inbox() == 0
    assert f.exists()  # left in place, not deleted
    assert db.latest_reading(demo_sensors["breakroom_cam"]) is None


def test_ingest_skips_unknown_sensor_dir(demo_sensors):
    f = _inbox_file("ghost_cam/shot.png")
    assert scheduler.ingest_inbox() == 0
    assert f.exists()
    for sid in demo_sensors.values():
        assert db.latest_reading(sid) is None


def test_ingest_skips_numeric_sensor_named_dir(demo_sensors):
    f = _inbox_file("keg_scale/shot.png")
    assert scheduler.ingest_inbox() == 0
    assert f.exists()
    assert db.latest_reading(demo_sensors["keg_scale"]) is None


def test_ingest_skips_nested_dirs_without_crash(demo_sensors):
    nested = _inbox_file("breakroom_cam/nested/deep.png")
    # a directory whose name looks like an image must not be treated as a file
    (Path(config.INBOX_DIR) / "breakroom_cam" / "fake.png.d").mkdir()
    assert scheduler.ingest_inbox() == 0
    assert nested.exists()
    assert db.latest_reading(demo_sensors["breakroom_cam"]) is None


def test_ingest_empty_or_missing_inbox(demo_sensors):
    # inbox dir does not exist at all
    assert not Path(config.INBOX_DIR).exists()
    assert scheduler.ingest_inbox() == 0
    # empty inbox, plus a stray file at inbox root (not in a sensor dir)
    Path(config.INBOX_DIR).mkdir()
    (Path(config.INBOX_DIR) / "stray.png").write_bytes(b"x")
    assert scheduler.ingest_inbox() == 0


# =========================================================================
# 2. scheduler.evaluate_due_rules
# =========================================================================

def test_only_active_rules_run(demo_sensors):
    sid = demo_sensors["keg_scale"]
    _mk_rule(sid, _threshold(), status="pending_confirm")
    _mk_rule(sid, _threshold(), status="disabled")
    db.create_reading(sid, "numeric", value=5)
    assert scheduler.evaluate_due_rules() == 0
    assert db.recent_evaluations() == []
    # activate one -> it runs
    active = _mk_rule(sid, _threshold())
    assert scheduler.evaluate_due_rules() == 1
    evs = db.recent_evaluations()
    assert len(evs) == 1 and evs[0]["rule_id"] == active["id"]


def test_rule_not_reevaluated_for_same_reading(demo_sensors):
    sid = demo_sensors["keg_scale"]
    _mk_rule(sid, _threshold())
    db.create_reading(sid, "numeric", value=5)
    assert scheduler.evaluate_due_rules() == 1
    assert scheduler.evaluate_due_rules() == 0  # nothing new
    assert len(db.recent_evaluations()) == 1
    # a new reading makes it due again
    db.create_reading(sid, "numeric", value=4)
    assert scheduler.evaluate_due_rules() == 1


def test_rule_with_no_readings_is_skipped(demo_sensors):
    _mk_rule(demo_sensors["keg_scale"], _threshold())
    assert scheduler.evaluate_due_rules() == 0
    assert db.recent_evaluations() == []


def test_rule_with_ghost_sensor_name_is_skipped(demo_sensors):
    # parsed_json names a sensor that is not in the DB
    _mk_rule(demo_sensors["keg_scale"], _threshold(sensor="renamed_scale"))
    db.create_reading(demo_sensors["keg_scale"], "numeric", value=5)
    assert scheduler.evaluate_due_rules() == 0


def test_corrupt_parsed_json_does_not_kill_cycle(demo_sensors):
    sid = demo_sensors["keg_scale"]
    db.create_rule(sid, "bad", "{{ not json at all", status="active")  # corrupt first
    good = _mk_rule(sid, _threshold())
    db.create_reading(sid, "numeric", value=5)
    assert scheduler.evaluate_due_rules() == 1  # good rule still ran
    assert db.recent_evaluations()[0]["rule_id"] == good["id"]


def test_schemaless_parsed_json_does_not_kill_cycle(demo_sensors):
    sid = demo_sensors["keg_scale"]
    db.create_rule(sid, "bad", "{}", status="active")  # valid JSON, no schema
    good = _mk_rule(sid, _threshold())
    db.create_reading(sid, "numeric", value=5)
    ran = scheduler.evaluate_due_rules()
    assert ran == 1
    assert db.recent_evaluations()[0]["rule_id"] == good["id"]


# =========================================================================
# 3. run_cycle end-to-end (mocked llm.ask_image)
# =========================================================================

def test_run_cycle_png_to_alert(demo_sensors, monkeypatch):
    monkeypatch.setattr(llm, "ask_image", lambda p, q, **kw: {
        "answer": "yes", "reason": "empty", "latency_ms": 5})
    rule = _mk_rule(demo_sensors["breakroom_cam"], _image_rule())
    f = _inbox_file("breakroom_cam/shot.png")
    out = scheduler.run_cycle()
    assert out["ingested"] == 1
    assert out["evaluated"] == 1
    assert not f.exists()
    rd = db.latest_reading(demo_sensors["breakroom_cam"])
    assert Path(rd["image_path"]).is_file()
    ev = db.last_evaluation_for_rule(rule["id"])
    assert ev["result"] == "triggered" and ev["reading_id"] == rd["id"]
    alert = db.last_alert_for_rule(rule["id"])
    assert alert is not None and alert["message"] == "img alert"
    # second cycle: nothing new, no double-evaluation
    out2 = scheduler.run_cycle()
    assert out2["ingested"] == 0 and out2["evaluated"] == 0


# =========================================================================
# 4. API edges
# =========================================================================

@pytest.fixture
def client():
    return TestClient(app)


def test_duplicate_sensor_409(demo_sensors, client):
    r = client.post("/api/sensors",
                    json={"name": "breakroom_cam", "kind": "image"})
    assert r.status_code == 409


def test_bad_sensor_kind_400(fresh_db, client):
    r = client.post("/api/sensors", json={"name": "x", "kind": "audio"})
    assert r.status_code == 400
    assert db.get_sensor_by_name("x") is None


def test_reading_unknown_sensor_404(fresh_db, client):
    r = client.post("/api/readings", json={"sensor": "nope", "value": 1})
    assert r.status_code == 404


def test_reading_for_image_sensor_400(demo_sensors, client):
    r = client.post("/api/readings", json={"sensor": "breakroom_cam", "value": 1})
    assert r.status_code == 400
    assert db.latest_reading(demo_sensors["breakroom_cam"]) is None


def test_voice_501(fresh_db, client):
    r = client.post("/api/voice", files={"file": ("a.wav", b"RIFFxxxx", "audio/wav")})
    assert r.status_code == 501


def test_alerts_ack_roundtrip(demo_sensors, client):
    rule = _mk_rule(demo_sensors["keg_scale"], _threshold())
    rd = db.get_reading(db.create_reading(demo_sensors["keg_scale"], "numeric", value=5))
    evaluator.evaluate(rule, rd)
    unacked = client.get("/api/alerts", params={"unacked": 1}).json()
    assert len(unacked) == 1 and unacked[0]["acknowledged"] == 0
    aid = unacked[0]["id"]
    assert client.post(f"/api/alerts/{aid}/ack").status_code == 200
    assert client.get("/api/alerts", params={"unacked": 1}).json() == []
    allrows = client.get("/api/alerts").json()
    assert len(allrows) == 1 and allrows[0]["acknowledged"] == 1


def test_ack_nonexistent_alert_404(fresh_db, client):
    assert client.post("/api/alerts/9999/ack").status_code == 404


def test_parse_with_no_sensors_400(fresh_db, client, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("LLM must not be called when no sensors exist")
    monkeypatch.setattr(llm, "parse_rule", boom)
    r = client.post("/api/rules/parse", json={"text": "watch stuff"})
    assert r.status_code == 400


def test_parse_empty_text_400(demo_sensors, client, monkeypatch):
    monkeypatch.setattr(llm, "parse_rule",
                        lambda *a, **kw: pytest.fail("LLM called on empty text"))
    r = client.post("/api/rules/parse", json={"text": "   "})
    assert r.status_code == 400


def test_confirm_disable_nonexistent_rule_404(fresh_db, client):
    assert client.post("/api/rules/9999/confirm").status_code == 404
    assert client.post("/api/rules/9999/disable").status_code == 404


def test_images_serves_legit_file(fresh_db, client):
    images = Path(config.IMAGES_DIR)
    images.mkdir(parents=True, exist_ok=True)
    (images / "ok.png").write_bytes(b"IMGDATA")
    r = client.get("/images/ok.png")
    assert r.status_code == 200
    assert r.content == b"IMGDATA"


def test_images_path_traversal_blocked(fresh_db, client):
    # secret lives one level ABOVE IMAGES_DIR (same dir as the sqlite db)
    secret = fresh_db / "secret.txt"
    secret.write_bytes(b"TOPSECRET")
    Path(config.IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    attempts = [
        "/images/..%2Fsecret.txt",
        "/images/%2e%2e%2fsecret.txt",
        "/images/..%2F..%2Fetc%2Fpasswd",
        "/images/../secret.txt",
        "/images/../../etc/passwd",
        "/images/..%5Csecret.txt",
        "/images/..%2Ftest.db",
    ]
    for url in attempts:
        r = client.get(url)
        assert r.status_code != 200, f"{url} leaked (status {r.status_code})"
        assert b"TOPSECRET" not in r.content
        assert b"root:" not in r.content


# =========================================================================
# 5. evaluator edge cases
# =========================================================================

def test_numeric_reading_with_none_value_writes_error_row(demo_sensors):
    rule = _mk_rule(demo_sensors["keg_scale"], _threshold())
    rd = db.get_reading(db.create_reading(demo_sensors["keg_scale"], "numeric", value=None))
    out = evaluator.evaluate(rule, rd)
    assert out["result"] == "error"
    assert out["alerted"] is False
    ev = db.last_evaluation_for_rule(rule["id"])
    assert ev["result"] == "error"
    assert ev["model_answer"] is None  # numeric path never involves the LLM
    assert db.last_alert_for_rule(rule["id"]) is None


def test_eq_threshold_with_float_value(demo_sensors):
    sid = demo_sensors["keg_scale"]
    rule = _mk_rule(sid, _threshold(op="eq", value=95.5))
    rd = db.get_reading(db.create_reading(sid, "numeric", value=95.5))
    assert evaluator.evaluate(rule, rd)["result"] == "triggered"
    rd2 = db.get_reading(db.create_reading(sid, "numeric", value=95.5000001))
    assert evaluator.evaluate(rule, rd2)["result"] == "ok"


def test_image_rule_dead_llm_endpoint_writes_error_no_alert(
        demo_sensors, tmp_path, monkeypatch):
    # Real llm.ask_image, but the endpoint is a dead localhost port: the
    # connection is refused instantly, ask_image returns 'unsure', evaluator
    # writes an error row and never alerts. No real network (127.0.0.1).
    monkeypatch.setattr(config, "MODEL_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setattr(config, "LLM_TIMEOUT_S", 2.0)
    img = tmp_path / "real.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    rule = _mk_rule(demo_sensors["breakroom_cam"], _image_rule())
    rd = db.get_reading(db.create_reading(
        demo_sensors["breakroom_cam"], "image", image_path=str(img)))
    out = evaluator.evaluate(rule, rd)
    assert out["result"] == "error"
    assert out["alerted"] is False
    ev = db.last_evaluation_for_rule(rule["id"])
    assert ev["result"] == "error"
    assert json.loads(ev["model_answer"])["answer"] == "unsure"
    assert db.last_alert_for_rule(rule["id"]) is None


def test_image_rule_missing_image_file_writes_error_row(demo_sensors, tmp_path):
    rule = _mk_rule(demo_sensors["breakroom_cam"], _image_rule())
    rd = db.get_reading(db.create_reading(
        demo_sensors["breakroom_cam"], "image",
        image_path=str(tmp_path / "does_not_exist.png")))
    out = evaluator.evaluate(rule, rd)  # must not raise
    assert out["result"] == "error"
    assert out["alerted"] is False
    ev = db.last_evaluation_for_rule(rule["id"])
    assert "error" in json.loads(ev["model_answer"])


def test_image_rule_reading_without_image_path(demo_sensors):
    rule = _mk_rule(demo_sensors["breakroom_cam"], _image_rule())
    rd = db.get_reading(db.create_reading(demo_sensors["breakroom_cam"], "image"))
    out = evaluator.evaluate(rule, rd)
    assert out["result"] == "error"
    assert out["alerted"] is False


# =========================================================================
# 6. rules.strip_fences fuzzing
# =========================================================================

from sentinel import rules as rules_mod  # noqa: E402


@pytest.mark.parametrize("raw", [
    '```json\n{"a": 1}\n```',                       # language tag
    '```\n{"a": 1}\n```',                            # no tag
    'Sure!\n```json\n{"a": 1}\n```\nHope it helps',  # prose around fence
    'Here you go: {"a": 1}',                          # prose before bare JSON
])
def test_strip_fences_simple_variants(raw):
    assert json.loads(rules_mod.strip_fences(raw)) == {"a": 1}


def test_strip_fences_nested_braces_in_prose():
    raw = 'Result follows: {"a": {"b": {"c": 2}}} '
    assert json.loads(rules_mod.strip_fences(raw)) == {"a": {"b": {"c": 2}}}


def test_strip_fences_trailing_prose_after_json():
    raw = '{"a": 1} Hope that helps!'
    assert json.loads(rules_mod.strip_fences(raw)) == {"a": 1}


def test_strip_fences_empty_and_garbage_do_not_crash():
    assert rules_mod.strip_fences("") == ""
    rules_mod.strip_fences("no braces here at all")  # must not raise


# =========================================================================
# 7. boolean readings via API + state_change through run_cycle
# =========================================================================

def test_boolean_reading_api_and_state_change_cycle(demo_sensors, client):
    rule = _mk_rule(demo_sensors["door_state"], {
        "sensor": "door_state", "modality": "boolean",
        "condition": {"type": "state_change", "from": False, "to": True},
        "action": {"type": "alert", "message": "Door opened"},
        "cooldown_minutes": 0,
    })
    r = client.post("/api/readings", json={"sensor": "door_state", "value": 0})
    assert r.status_code == 200
    out1 = client.post("/api/cycle").json()
    assert out1["evaluated"] == 1
    assert db.last_evaluation_for_rule(rule["id"])["result"] == "ok"
    assert db.last_alert_for_rule(rule["id"]) is None

    r = client.post("/api/readings", json={"sensor": "door_state", "value": 1})
    assert r.status_code == 200
    out2 = client.post("/api/cycle").json()
    assert out2["evaluated"] == 1
    assert db.last_evaluation_for_rule(rule["id"])["result"] == "triggered"
    alert = db.last_alert_for_rule(rule["id"])
    assert alert is not None and alert["message"] == "Door opened"

    # same reading again -> not re-evaluated, no duplicate alert
    out3 = client.post("/api/cycle").json()
    assert out3["evaluated"] == 0
    assert len([a for a in db.list_alerts() if a["rule_id"] == rule["id"]]) == 1
