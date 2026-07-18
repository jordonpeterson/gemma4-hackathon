# Sentinel

Natural-language rule engine for arbitrary sensors, running entirely on one
older Mac laptop, CPU-only, no cloud, no API keys.

Type (or eventually speak) a rule like *"alert me when the coke box in the
break room is empty"*. Sentinel parses it into a structured rule with a local
Gemma model, asks you to confirm the parse, then periodically evaluates sensor
inputs — images through the vision model, numbers through plain Python — and
fires macOS notifications when rules trigger.

```
 voice/text ──▶ Gemma (llama.cpp) ──▶ structured rule ──▶ you confirm
                                                              │
 inbox images ─┐                                              ▼
 HTTP readings ┴──▶ scheduler ──▶ evaluator ──▶ alerts (macOS notification + log)
```

## Requirements

- macOS (older Intel or early Apple Silicon is fine), 8–16 GB RAM
- Python 3.11+
- [llama.cpp](https://github.com/ggml-org/llama.cpp): `brew install llama.cpp`

## Quick start

```bash
# 1. Start the model server (first run auto-downloads the GGUF, a few GB)
./scripts/run_model.sh

# 2. In another terminal:
pip install -r requirements.txt
python scripts/seed_demo.py     # 2 demo sensors + 3 sample images in the inbox
python -m sentinel              # API + scheduler on http://127.0.0.1:8000

# 3. Open http://127.0.0.1:8000
```

### Model

Target model is **Gemma 4 E2B (Q4_K_M GGUF)**. No multimodal GGUF of it is
published at the time of writing, so the default is the spec's fallback:
**Gemma 3 4B-it** with its vision projector, pulled automatically from
[`ggml-org/gemma-3-4b-it-GGUF`](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF)
by `llama-server -hf`. Everything model-facing lives behind `sentinel/llm.py`,
so swapping models is one env var:

```bash
MODEL_HF=ggml-org/gemma-4-e2b-it-GGUF ./scripts/run_model.sh   # when it exists
```

## The demo (definition of done)

1. In the UI **Teach** panel, type: `alert me when the coke box in the break
   room is empty` → Sentinel shows the parsed rule ("Watch **breakroom_cam**.
   Alert when: *Is the coke box empty?* …") → click **Confirm**. Rules are
   never auto-activated; the confirm step is the mis-parse safety net.
2. Drop a photo of an empty box into `inbox/breakroom_cam/` (seed_demo already
   put one there). Within one poll cycle (default 5 min — or `curl -X POST
   localhost:8000/api/cycle` to run one immediately) you get a macOS
   notification.
3. Numeric rules never touch the LLM. Teach "alert me when the keg drops below
   15 pounds", confirm, then:

   ```bash
   curl -X POST localhost:8000/api/readings \
     -H 'Content-Type: application/json' \
     -d '{"sensor": "keg_scale", "value": 12}'
   curl -X POST localhost:8000/api/cycle
   ```

   The alert fires with `model_answer IS NULL` in the evaluations table.

## How it works

| Module | Job |
| --- | --- |
| `sentinel/llm.py` | Only file that talks to llama-server. Two calls: `parse_rule` (text → rule JSON, one retry on invalid output) and `ask_image` (yes/no/unsure about an image). Serialized with a lock; 180 s timeout because CPU. |
| `sentinel/rules.py` | Canonical rule schema (pydantic), fence-stripping, fuzzy sensor matching (`difflib`, cutoff 0.6), plain-English summaries, pending_confirm flow. |
| `sentinel/evaluator.py` | The important split: numeric/boolean rules are pure Python and **never call the LLM**; image rules make one vision call. `unsure` → error row for review, never an alert. `active_hours` and `cooldown_minutes` (default 240) gate alert creation. Every evaluation is logged. |
| `sentinel/scheduler.py` | Background thread. Every `SENTINEL_POLL_SECONDS` (default 300): ingest `inbox/<sensor>/` images → evaluate active rules with new readings → alert. |
| `sentinel/alerts.py` | Alert row + macOS notification via `osascript`. `send_email` is a deliberate `NotImplementedError` stub. |
| `sentinel/api.py` | FastAPI routes + the single-page admin UI (`sentinel/static/index.html`, vanilla JS, no build step). |

### Rule schema

```json
{
  "sensor": "breakroom_cam",
  "modality": "image | numeric | boolean",
  "condition": {
    "type": "visual_question | threshold | state_change",
    "question": "Is the coke box empty?",
    "operator": "lt | gt | eq", "value": 15,
    "from": null, "to": null
  },
  "action": {"type": "alert", "message": "Restock cokes in break room"},
  "active_hours": {"start": "00:00", "end": "23:59"},
  "cooldown_minutes": 240
}
```

### API

```
POST /api/rules/parse        {text} → pending_confirm rule + summary
POST /api/rules/{id}/confirm
POST /api/rules/{id}/disable
GET  /api/rules
GET  /api/sensors            POST /api/sensors {name, kind, location}
POST /api/readings           {sensor, value}
POST /api/voice              501 for now (seam kept for audio input)
GET  /api/alerts?unacked=1   POST /api/alerts/{id}/ack
GET  /api/evaluations
GET  /api/health
POST /api/cycle              run one ingest+evaluate cycle now (demo helper)
```

## Configuration

All env-overridable, see `sentinel/config.py`: `SENTINEL_DB`,
`SENTINEL_MODEL_ENDPOINT` (default `http://localhost:8080`),
`SENTINEL_POLL_SECONDS`, `SENTINEL_INBOX`, `SENTINEL_IMAGES`,
`SENTINEL_HOST`/`SENTINEL_PORT`, `SENTINEL_LLM_TIMEOUT`.

## Tests

```bash
pytest                          # no model needed, all mocked
SENTINEL_LIVE_TESTS=1 pytest    # includes live llama-server integration test
```

## Non-goals (MVP)

Real sensor pairing, LoRa, auth/multi-user, cloud sync, mobile app, model
fine-tuning, video, Docker.
