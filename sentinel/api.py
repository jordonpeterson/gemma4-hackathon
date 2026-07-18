"""FastAPI routes + static admin UI."""
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sentinel import config, db, llm, rules, scheduler

app = FastAPI(title="Sentinel", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"


class ParseRequest(BaseModel):
    text: str


class SensorCreate(BaseModel):
    name: str
    kind: str  # image | numeric | boolean
    location: str = ""


class ReadingCreate(BaseModel):
    sensor: str
    value: float


# ---------- rules ----------

@app.post("/api/rules/parse")
def parse_rule(req: ParseRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty instruction")
    known = [s["name"] for s in db.list_sensors()]
    if not known:
        raise HTTPException(400, "no sensors defined yet — create a sensor first")
    parsed = llm.parse_rule(text, known)
    if "error" in parsed:
        status = 502 if parsed["error"] in ("llm_unavailable",) else 422
        return JSONResponse(status_code=status, content=parsed)
    try:
        row = rules.create_pending_rule(text, parsed)
    except rules.ModalityMismatch as exc:
        return JSONResponse(status_code=422,
                            content={"error": "modality_mismatch", "detail": str(exc)})
    return row  # includes parsed rule + human-readable summary, status pending_confirm


@app.post("/api/rules/{rule_id}/confirm")
def confirm_rule(rule_id: int):
    row = db.get_rule(rule_id)
    if row is None:
        raise HTTPException(404, "rule not found")
    if row["status"] == "active":
        return {"ok": True, "status": "active"}
    db.set_rule_status(rule_id, "active")
    return {"ok": True, "status": "active"}


@app.post("/api/rules/{rule_id}/disable")
def disable_rule(rule_id: int):
    if db.get_rule(rule_id) is None:
        raise HTTPException(404, "rule not found")
    db.set_rule_status(rule_id, "disabled")
    return {"ok": True, "status": "disabled"}


@app.get("/api/rules")
def get_rules():
    out = []
    for r in db.list_rules():
        try:
            parsed = json.loads(r["parsed_json"])
            r["parsed"] = parsed
            r["summary"] = rules.summarize(parsed)
        except Exception:
            r["parsed"] = None
            r["summary"] = "(unparseable rule — disable it)"
        out.append(r)
    return out


# ---------- sensors ----------

@app.get("/api/sensors")
def get_sensors():
    out = []
    for s in db.list_sensors():
        s["latest_reading"] = db.latest_reading(s["id"])
        out.append(s)
    return out


@app.post("/api/sensors")
def create_sensor(req: SensorCreate):
    if req.kind not in ("image", "numeric", "boolean"):
        raise HTTPException(400, "kind must be image|numeric|boolean")
    if db.get_sensor_by_name(req.name):
        raise HTTPException(409, f"sensor {req.name!r} already exists")
    sensor_id = db.create_sensor(req.name, req.kind, req.location)
    return db.get_sensor(sensor_id)


# ---------- readings ----------

@app.post("/api/readings")
def create_reading(req: ReadingCreate):
    sensor = db.get_sensor_by_name(req.sensor)
    if sensor is None:
        raise HTTPException(404, f"unknown sensor {req.sensor!r}")
    if sensor["kind"] == "image":
        raise HTTPException(400, "image readings arrive via the inbox directory")
    reading_id = db.create_reading(sensor["id"], sensor["kind"], value=req.value)
    return {"ok": True, "reading_id": reading_id}


# ---------- voice (seam only) ----------

@app.post("/api/voice")
def voice(file: UploadFile):
    # Seam kept for when the local model gains audio input. For MVP: text only.
    raise HTTPException(501, "use text for now")


# ---------- alerts ----------

@app.get("/api/alerts")
def get_alerts(unacked: int = 0):
    return db.list_alerts(unacked_only=bool(unacked))


@app.post("/api/alerts/{alert_id}/ack")
def ack_alert(alert_id: int):
    if not db.ack_alert(alert_id):
        raise HTTPException(404, "alert not found")
    return {"ok": True}


# ---------- evaluations (audit trail for the UI) ----------

@app.get("/api/evaluations")
def get_evaluations(limit: int = 50):
    return db.recent_evaluations(limit=min(limit, 200))


# ---------- ops ----------

@app.get("/api/health")
def health():
    db_ok = True
    try:
        db.list_sensors()
    except Exception:
        db_ok = False
    return {
        "llama_server": llm.health(),
        "db": db_ok,
        "last_cycle": scheduler.last_cycle_time(),
        "poll_seconds": config.POLL_SECONDS,
    }


@app.post("/api/cycle")
def run_cycle_now():
    """Manual poke — run one ingest+evaluate cycle immediately (handy for demos)."""
    return scheduler.run_cycle()


# ---------- static ----------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/images/{name}")
def image(name: str):
    # Serve ingested images for UI thumbnails; no traversal.
    p = (Path(config.IMAGES_DIR) / Path(name).name).resolve()
    if not p.is_file() or Path(config.IMAGES_DIR).resolve() not in p.parents:
        raise HTTPException(404, "not found")
    return FileResponse(p)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
