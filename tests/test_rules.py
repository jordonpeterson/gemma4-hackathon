"""Rule parsing/validation tests. No model required — llm internals mocked."""
import json
import os

import pytest
from fastapi.testclient import TestClient

from sentinel import db, llm, rules
from sentinel.api import app

VALID_IMAGE_RULE = {
    "sensor": "breakroom_cam",
    "modality": "image",
    "condition": {"type": "visual_question", "question": "Is the coke box empty?"},
    "action": {"type": "alert", "message": "Restock cokes"},
}

VALID_THRESHOLD_RULE = {
    "sensor": "keg_scale",
    "modality": "numeric",
    "condition": {"type": "threshold", "operator": "lt", "value": 15},
    "action": {"type": "alert", "message": "Keg low"},
}

KNOWN = ["breakroom_cam", "keg_scale", "door_state"]


# ---------- schema validation ----------

def test_valid_rule_passes_and_gets_defaults():
    out = rules.validate_parsed(VALID_IMAGE_RULE, KNOWN)
    assert out["sensor"] == "breakroom_cam"
    assert out["cooldown_minutes"] == 240
    assert out["active_hours"] == {"start": "00:00", "end": "23:59"}


def test_threshold_requires_operator_and_value():
    bad = json.loads(json.dumps(VALID_THRESHOLD_RULE))
    del bad["condition"]["operator"]
    with pytest.raises(Exception):
        rules.validate_parsed(bad, KNOWN)


def test_visual_question_requires_image_modality():
    bad = json.loads(json.dumps(VALID_IMAGE_RULE))
    bad["modality"] = "numeric"
    with pytest.raises(Exception):
        rules.validate_parsed(bad, KNOWN)


def test_bad_active_hours_rejected():
    bad = json.loads(json.dumps(VALID_IMAGE_RULE))
    bad["active_hours"] = {"start": "9am", "end": "17:00"}
    with pytest.raises(Exception):
        rules.validate_parsed(bad, KNOWN)


# ---------- fuzzy sensor matching ----------

def test_fuzzy_sensor_match_normalizes_name():
    rule = json.loads(json.dumps(VALID_IMAGE_RULE))
    rule["sensor"] = "breakroom cam"  # close but not exact
    out = rules.validate_parsed(rule, KNOWN)
    assert out["sensor"] == "breakroom_cam"


def test_unknown_sensor_returns_structured_error():
    rule = json.loads(json.dumps(VALID_IMAGE_RULE))
    rule["sensor"] = "parking_lot_drone"
    out = rules.validate_parsed(rule, KNOWN)
    assert out["error"] == "unknown_sensor"
    assert out["candidates"] == KNOWN


# ---------- fence stripping ----------

def test_strip_fences_handles_markdown_and_prose():
    fenced = "```json\n{\"a\": 1}\n```"
    assert json.loads(rules.strip_fences(fenced)) == {"a": 1}
    prose = "Sure! Here is the rule: {\"a\": 1} Hope that helps."
    assert json.loads(rules.strip_fences(prose)) == {"a": 1}


# ---------- llm.parse_rule retry logic (mock the chat layer) ----------

def test_parse_rule_retries_once_then_succeeds(monkeypatch):
    outputs = ["this is not json at all", json.dumps(VALID_IMAGE_RULE)]
    calls = []
    monkeypatch.setattr(llm, "_chat", lambda msgs, **kw: (calls.append(1), outputs[len(calls) - 1])[1])
    out = llm.parse_rule("alert when coke box empty", KNOWN)
    assert len(calls) == 2
    assert out["sensor"] == "breakroom_cam"


def test_parse_rule_retry_then_fail(monkeypatch):
    monkeypatch.setattr(llm, "_chat", lambda msgs, **kw: "still { not json")
    out = llm.parse_rule("gibberish", KNOWN)
    assert out["error"] == "parse_failed"
    assert "detail" in out


def test_parse_rule_llm_down(monkeypatch):
    def boom(msgs, **kw):
        raise ConnectionError("refused")
    monkeypatch.setattr(llm, "_chat", boom)
    out = llm.parse_rule("anything", KNOWN)
    assert out["error"] == "llm_unavailable"


# ---------- pending_confirm flow through the API ----------

def test_pending_confirm_flow(demo_sensors, monkeypatch):
    monkeypatch.setattr(llm, "parse_rule", lambda text, known: dict(VALID_THRESHOLD_RULE))
    client = TestClient(app)

    r = client.post("/api/rules/parse", json={"text": "alert below 15 lbs"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "pending_confirm"
    assert "keg_scale" in body["summary"]

    # Not active yet: never auto-activate.
    assert db.get_rule(body["id"])["status"] == "pending_confirm"

    r = client.post(f"/api/rules/{body['id']}/confirm")
    assert r.status_code == 200
    assert db.get_rule(body["id"])["status"] == "active"

    r = client.post(f"/api/rules/{body['id']}/disable")
    assert r.status_code == 200
    assert db.get_rule(body["id"])["status"] == "disabled"


def test_parse_endpoint_surfaces_unknown_sensor(demo_sensors, monkeypatch):
    monkeypatch.setattr(llm, "parse_rule",
                        lambda text, known: {"error": "unknown_sensor", "candidates": known})
    client = TestClient(app)
    r = client.post("/api/rules/parse", json={"text": "watch the drone"})
    assert r.status_code == 422
    assert r.json()["error"] == "unknown_sensor"


# ---------- live integration (opt-in) ----------

@pytest.mark.skipif(os.environ.get("SENTINEL_LIVE_TESTS") != "1",
                    reason="set SENTINEL_LIVE_TESTS=1 with llama-server running")
def test_parse_rule_live():
    out = llm.parse_rule("alert me when the coke box in the break room is empty", KNOWN)
    assert out.get("sensor") == "breakroom_cam"
    assert out["condition"]["type"] == "visual_question"


# ---------- modality vs sensor-kind guard ----------

def test_modality_mismatch_rejected(demo_sensors):
    bad = {
        "sensor": "breakroom_cam", "modality": "numeric",
        "condition": {"type": "threshold", "operator": "lt", "value": 5},
        "action": {"type": "alert", "message": "x"},
    }
    parsed = rules.validate_parsed(bad, KNOWN)
    with pytest.raises(rules.ModalityMismatch):
        rules.create_pending_rule("bad", parsed)


def test_modality_mismatch_via_api_is_422(demo_sensors, monkeypatch):
    bad = {
        "sensor": "keg_scale", "modality": "image",
        "condition": {"type": "visual_question", "question": "Empty?"},
        "action": {"type": "alert", "message": "x"},
        "active_hours": {"start": "00:00", "end": "23:59"},
        "cooldown_minutes": 240,
    }
    monkeypatch.setattr(llm, "parse_rule", lambda text, known: dict(bad))
    client = TestClient(app)
    r = client.post("/api/rules/parse", json={"text": "watch keg photo"})
    assert r.status_code == 422
    assert r.json()["error"] == "modality_mismatch"


def test_boolean_rule_on_numeric_sensor_allowed(demo_sensors):
    ok = {
        "sensor": "keg_scale", "modality": "boolean",
        "condition": {"type": "state_change", "to": True},
        "action": {"type": "alert", "message": "x"},
    }
    parsed = rules.validate_parsed(ok, KNOWN)
    row = rules.create_pending_rule("ok", parsed)
    assert row["status"] == "pending_confirm"


# ---------- context reaches the vision prompt ----------

def test_ask_image_injects_context_into_system_prompt(tmp_path, monkeypatch):
    captured = {}

    def fake_chat(messages, **kw):
        captured["messages"] = messages
        return '{"answer": "no", "reason": "ok"}'

    monkeypatch.setattr(llm, "_chat", fake_chat)
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8fake")
    out = llm.ask_image(str(img), "Is the box empty?",
                        context="Camera 2 monitors the 2nd floor break room.")
    assert out["answer"] == "no"
    system = captured["messages"][0]["content"]
    assert "Camera 2 monitors the 2nd floor break room." in system


def test_ask_image_without_context_unchanged(tmp_path, monkeypatch):
    captured = {}

    def fake_chat(messages, **kw):
        captured["messages"] = messages
        return '{"answer": "no", "reason": "ok"}'

    monkeypatch.setattr(llm, "_chat", fake_chat)
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8fake")
    llm.ask_image(str(img), "Is the box empty?")
    assert "Camera context" not in captured["messages"][0]["content"]
