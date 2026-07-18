"""Canonical rule schema, validation, fuzzy sensor matching, and the
pending_confirm flow.

This module never talks to the LLM (llm.py imports *us*, not the other
way round) and never auto-activates a rule — activation only happens via
the explicit /confirm endpoint. That is the mis-parse safety net.
"""
import difflib
import json
import re
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from sentinel import config, db

_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class Condition(BaseModel):
    type: Literal["visual_question", "threshold", "state_change"]
    question: Optional[str] = None                       # visual_question
    operator: Optional[Literal["lt", "gt", "eq"]] = None  # threshold
    value: Optional[float] = None                         # threshold
    from_: Optional[Union[float, bool]] = Field(default=None, alias="from")
    to: Optional[Union[float, bool]] = None               # state_change

    model_config = {"populate_by_name": True}


class Action(BaseModel):
    type: Literal["alert"] = "alert"
    message: str


class ActiveHours(BaseModel):
    start: str = "00:00"
    end: str = "23:59"

    @field_validator("start", "end")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        if not _HHMM.match(v):
            raise ValueError(f"time must be HH:MM (24h), got {v!r}")
        return v


class Rule(BaseModel):
    sensor: str
    modality: Literal["image", "numeric", "boolean"]
    condition: Condition
    action: Action
    active_hours: ActiveHours = Field(default_factory=ActiveHours)
    cooldown_minutes: int = Field(default=config.DEFAULT_COOLDOWN_MINUTES, ge=0)

    @model_validator(mode="after")
    def _cross_checks(self) -> "Rule":
        c = self.condition
        if c.type == "visual_question":
            if self.modality != "image":
                raise ValueError("visual_question conditions require modality 'image'")
            if not c.question:
                raise ValueError("visual_question conditions require a 'question'")
        elif c.type == "threshold":
            if self.modality != "numeric":
                raise ValueError("threshold conditions require modality 'numeric'")
            if c.operator is None or c.value is None:
                raise ValueError("threshold conditions require 'operator' and 'value'")
        elif c.type == "state_change":
            if self.modality == "image":
                raise ValueError("state_change conditions require modality 'numeric' or 'boolean'")
            if c.to is None:
                raise ValueError("state_change conditions require 'to'")
        return self


def strip_fences(text: str) -> str:
    """Defensively remove ```json ... ``` fences and surrounding chatter."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    # If there is still prose around the JSON, grab the outermost object.
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    return text


def match_sensor(name: str, known_sensors: list[str]) -> Optional[str]:
    """Fuzzy-match a sensor name against known sensors. Returns the canonical
    name or None."""
    if name in known_sensors:
        return name
    matches = difflib.get_close_matches(
        name, known_sensors, n=1, cutoff=config.FUZZY_SENSOR_CUTOFF
    )
    return matches[0] if matches else None


def validate_parsed(data: dict, known_sensors: list[str]) -> dict:
    """Validate a raw parsed dict against the canonical schema.

    Returns the canonical rule dict (sensor name normalized to the fuzzy
    match). Raises ValueError with a readable message on schema problems.
    Returns a structured error dict {"error": "unknown_sensor", ...} when
    the sensor cannot be matched — that is a user problem, not a retryable
    model problem.
    """
    rule = Rule.model_validate(data)  # raises pydantic.ValidationError
    canonical = match_sensor(rule.sensor, known_sensors)
    if canonical is None:
        return {"error": "unknown_sensor", "candidates": known_sensors}
    rule.sensor = canonical
    return json.loads(rule.model_dump_json(by_alias=True))


def summarize(rule: dict) -> str:
    """Plain-English confirmation line for the UI."""
    c = rule["condition"]
    if c["type"] == "visual_question":
        cond = f"“{c['question']}” is true"
    elif c["type"] == "threshold":
        op = {"lt": "below", "gt": "above", "eq": "equal to"}[c["operator"]]
        cond = f"value is {op} {c['value']:g}"
    else:
        frm = c.get("from")
        cond = (f"state changes from {frm} to {c['to']}" if frm is not None
                else f"state becomes {c['to']}")
    hours = rule.get("active_hours") or {}
    hours_txt = ""
    if hours and (hours.get("start", "00:00"), hours.get("end", "23:59")) != ("00:00", "23:59"):
        hours_txt = f" Active {hours['start']}–{hours['end']}."
    cd = rule.get("cooldown_minutes", config.DEFAULT_COOLDOWN_MINUTES)
    cd_txt = f"{cd / 60:g} h" if cd >= 60 else f"{cd} min"
    return (f"Watch **{rule['sensor']}**. Alert when: *{cond}*."
            f" Cooldown {cd_txt}.{hours_txt} Correct?")


def create_pending_rule(raw_instruction: str, parsed: dict) -> dict:
    """Persist a validated rule as pending_confirm. Returns the DB row +
    summary."""
    sensor = db.get_sensor_by_name(parsed["sensor"])
    if sensor is None:  # should not happen after validate_parsed
        raise ValueError(f"sensor {parsed['sensor']!r} not found")
    rule_id = db.create_rule(sensor["id"], raw_instruction, json.dumps(parsed))
    row = db.get_rule(rule_id)
    row["parsed"] = parsed
    row["summary"] = summarize(parsed)
    return row
