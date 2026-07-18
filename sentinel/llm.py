"""Single client for llama-server (OpenAI-compatible, localhost).

Exactly two LLM-calling functions:
  parse_rule(text, known_sensors)  -> canonical rule dict | {"error": ...}
  ask_image(image_path, question)  -> {"answer": "yes|no|unsure", "reason": str}
(plus health(), a non-LLM reachability probe for /api/health).

CPU inference is slow and llama-server handles one request at a time well,
so every call is serialized behind a module-level lock (scheduler and API
share it).
"""
import base64
import json
import mimetypes
import threading
import time
from typing import Optional

import httpx
from pydantic import ValidationError

from sentinel import config, rules

_LOCK = threading.Lock()

_PARSE_SYSTEM = """You convert a user's spoken instruction into ONE JSON rule.

Output ONLY a JSON object. No markdown fences, no commentary.

Schema:
{
  "sensor": "<one of the known sensor names>",
  "modality": "image" | "numeric" | "boolean",
  "condition": {
    "type": "visual_question" | "threshold" | "state_change",
    "question": "<yes/no question about the image>",   // visual_question only
    "operator": "lt" | "gt" | "eq",                     // threshold only
    "value": <number>,                                  // threshold only
    "from": <number|bool|null>, "to": <number|bool>     // state_change only
  },
  "action": {"type": "alert", "message": "<short alert text>"},
  "active_hours": {"start": "HH:MM", "end": "HH:MM"},
  "cooldown_minutes": <int, default 240>
}

Known sensors: {SENSORS}

Rules of thumb:
- Image sensors -> visual_question with a yes/no question phrased so YES means "alert".
- Numeric comparisons ("below 15", "over 90") -> threshold with lt/gt/eq.
- "when X turns on/off", "goes from A to B" -> state_change.
- Omit active_hours unless the user gives hours. Omit cooldown_minutes unless given.

Examples:

Instruction: "alert me when the coke box in the break room is empty"
{"sensor": "breakroom_cam", "modality": "image", "condition": {"type": "visual_question", "question": "Is the coke box empty?"}, "action": {"type": "alert", "message": "Restock cokes in break room"}}

Instruction: "tell me if the keg drops below 15 pounds"
{"sensor": "keg_scale", "modality": "numeric", "condition": {"type": "threshold", "operator": "lt", "value": 15}, "action": {"type": "alert", "message": "Keg below 15 lbs — order a new one"}}

Instruction: "let me know when the compressor switches off"
{"sensor": "compressor_state", "modality": "boolean", "condition": {"type": "state_change", "from": true, "to": false}, "action": {"type": "alert", "message": "Compressor switched off"}}
"""

_IMAGE_SYSTEM = """You answer ONE yes/no question about the attached image.

Output ONLY a JSON object, no markdown fences:
{"answer": "yes" | "no" | "unsure", "reason": "<one short sentence>"}

Answer "unsure" if the image is unclear or does not show what the question
asks about. Never guess."""


def _chat(messages: list[dict], max_tokens: int = 512) -> str:
    """Serialized call to llama-server. Returns assistant text."""
    payload = {
        "model": config.MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    with _LOCK:
        resp = httpx.post(
            f"{config.MODEL_ENDPOINT}/v1/chat/completions",
            json=payload,
            timeout=config.LLM_TIMEOUT_S,
        )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def health() -> bool:
    try:
        r = httpx.get(f"{config.MODEL_ENDPOINT}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def parse_rule(text: str, known_sensors: list[str]) -> dict:
    """Instruction -> canonical rule dict, or {"error": ..., ...}.

    One retry with the validation error appended; then give up.
    """
    system = _PARSE_SYSTEM.replace("{SENSORS}", ", ".join(known_sensors) or "(none)")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Instruction: {text}"},
    ]
    last_error: Optional[str] = None
    for attempt in range(2):
        if last_error is not None:
            messages.append({
                "role": "user",
                "content": (f"Your previous output was invalid: {last_error}. "
                            "Output ONLY the corrected JSON object."),
            })
        try:
            raw = _chat(messages)
        except Exception as exc:
            return {"error": "llm_unavailable", "detail": str(exc)}
        messages.append({"role": "assistant", "content": raw})
        try:
            data = json.loads(rules.strip_fences(raw))
        except json.JSONDecodeError as exc:
            last_error = f"not valid JSON ({exc})"
            continue
        try:
            result = rules.validate_parsed(data, known_sensors)
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue
        return result  # canonical rule, or {"error": "unknown_sensor", ...}
    return {"error": "parse_failed", "detail": last_error}


def ask_image(image_path: str, question: str, context: str = "") -> dict:
    """Ask a yes/no question about an image.

    `context` is free text describing what the camera watches ("Fixed camera
    monitoring the snack wall in the 2nd floor break room") — it's appended
    to the system prompt so the model interprets ambiguous frames correctly.

    Always returns {"answer": "yes"|"no"|"unsure", "reason": str,
    "latency_ms": int}. Anything unparseable maps to "unsure".
    """
    system = _IMAGE_SYSTEM
    if context:
        system += f"\n\nCamera context: {context.strip()}"
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": question},
        ]},
    ]
    start = time.monotonic()
    try:
        raw = _chat(messages, max_tokens=128)
    except Exception as exc:
        latency = int((time.monotonic() - start) * 1000)
        return {"answer": "unsure", "reason": f"llm error: {exc}", "latency_ms": latency}
    latency = int((time.monotonic() - start) * 1000)
    try:
        data = json.loads(rules.strip_fences(raw))
        answer = str(data.get("answer", "")).strip().lower()
        if answer not in ("yes", "no", "unsure"):
            raise ValueError(f"bad answer {answer!r}")
        return {"answer": answer,
                "reason": str(data.get("reason", "")),
                "latency_ms": latency}
    except Exception:
        return {"answer": "unsure", "reason": f"unparseable model output: {raw[:200]}",
                "latency_ms": latency}
