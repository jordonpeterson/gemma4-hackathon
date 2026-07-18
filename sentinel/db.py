"""SQLite schema + typed accessors.

All timestamps are set by the DB layer (SQLite `datetime('now')`, UTC).
The LLM never generates timestamps.

Connections are opened per call: cheap for SQLite, and it keeps the module
safe to use from both the API thread-pool and the scheduler thread.
"""
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from sentinel import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensors(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('image','numeric','boolean')),
  location TEXT,
  context TEXT,                -- free text injected into vision prompts,
                               -- e.g. "Fixed camera watching the snack wall
                               -- in the 2nd floor break room"
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS rules(
  id INTEGER PRIMARY KEY,
  sensor_id INTEGER NOT NULL REFERENCES sensors(id),
  raw_instruction TEXT NOT NULL,
  parsed_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending_confirm','active','disabled')),
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS readings(
  id INTEGER PRIMARY KEY,
  sensor_id INTEGER NOT NULL REFERENCES sensors(id),
  kind TEXT NOT NULL CHECK(kind IN ('image','numeric','boolean')),
  value REAL,
  image_path TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS evaluations(
  id INTEGER PRIMARY KEY,
  rule_id INTEGER NOT NULL REFERENCES rules(id),
  reading_id INTEGER NOT NULL REFERENCES readings(id),
  result TEXT NOT NULL CHECK(result IN ('triggered','ok','error')),
  model_answer TEXT,
  latency_ms INTEGER,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY,
  rule_id INTEGER NOT NULL REFERENCES rules(id),
  evaluation_id INTEGER NOT NULL REFERENCES evaluations(id),
  message TEXT NOT NULL,
  acknowledged INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now'))
);
"""


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Open, commit-on-success, and ALWAYS close (macOS default ulimit -n is
    256 — leaked fds add up)."""
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # Lightweight migration for DBs created before sensors.context existed.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(sensors)")}
        if "context" not in cols:
            conn.execute("ALTER TABLE sensors ADD COLUMN context TEXT")


def _rows(cur) -> list[dict[str, Any]]:
    return [dict(r) for r in cur.fetchall()]


def _row(cur) -> Optional[dict[str, Any]]:
    r = cur.fetchone()
    return dict(r) if r else None


# ---------- sensors ----------

def create_sensor(name: str, kind: str, location: str = "",
                  context: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sensors(name, kind, location, context) VALUES (?,?,?,?)",
            (name, kind, location, context),
        )
        return cur.lastrowid


def list_sensors() -> list[dict]:
    with get_conn() as conn:
        return _rows(conn.execute("SELECT * FROM sensors ORDER BY id"))


def get_sensor_by_name(name: str) -> Optional[dict]:
    with get_conn() as conn:
        return _row(conn.execute("SELECT * FROM sensors WHERE name = ?", (name,)))


def get_sensor(sensor_id: int) -> Optional[dict]:
    with get_conn() as conn:
        return _row(conn.execute("SELECT * FROM sensors WHERE id = ?", (sensor_id,)))


# ---------- rules ----------

def create_rule(sensor_id: int, raw_instruction: str, parsed_json: str,
                status: str = "pending_confirm") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO rules(sensor_id, raw_instruction, parsed_json, status) VALUES (?,?,?,?)",
            (sensor_id, raw_instruction, parsed_json, status),
        )
        return cur.lastrowid


def get_rule(rule_id: int) -> Optional[dict]:
    with get_conn() as conn:
        return _row(conn.execute("SELECT * FROM rules WHERE id = ?", (rule_id,)))


def list_rules(status: Optional[str] = None) -> list[dict]:
    with get_conn() as conn:
        if status:
            cur = conn.execute("SELECT * FROM rules WHERE status = ? ORDER BY id", (status,))
        else:
            cur = conn.execute("SELECT * FROM rules ORDER BY id")
        return _rows(cur)


def set_rule_status(rule_id: int, status: str) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE rules SET status = ? WHERE id = ?", (status, rule_id))


# ---------- readings ----------

def create_reading(sensor_id: int, kind: str, value: Optional[float] = None,
                   image_path: Optional[str] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO readings(sensor_id, kind, value, image_path) VALUES (?,?,?,?)",
            (sensor_id, kind, value, image_path),
        )
        return cur.lastrowid


def get_reading(reading_id: int) -> Optional[dict]:
    with get_conn() as conn:
        return _row(conn.execute("SELECT * FROM readings WHERE id = ?", (reading_id,)))


def latest_reading(sensor_id: int) -> Optional[dict]:
    with get_conn() as conn:
        return _row(conn.execute(
            "SELECT * FROM readings WHERE sensor_id = ? ORDER BY id DESC LIMIT 1",
            (sensor_id,),
        ))


def previous_reading(sensor_id: int, before_reading_id: int) -> Optional[dict]:
    """The reading immediately before `before_reading_id` for this sensor."""
    with get_conn() as conn:
        return _row(conn.execute(
            "SELECT * FROM readings WHERE sensor_id = ? AND id < ? ORDER BY id DESC LIMIT 1",
            (sensor_id, before_reading_id),
        ))


# ---------- evaluations ----------

def create_evaluation(rule_id: int, reading_id: int, result: str,
                      model_answer: Optional[str] = None,
                      latency_ms: Optional[int] = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO evaluations(rule_id, reading_id, result, model_answer, latency_ms)"
            " VALUES (?,?,?,?,?)",
            (rule_id, reading_id, result, model_answer, latency_ms),
        )
        return cur.lastrowid


def last_evaluation_for_rule(rule_id: int) -> Optional[dict]:
    with get_conn() as conn:
        return _row(conn.execute(
            "SELECT * FROM evaluations WHERE rule_id = ? ORDER BY id DESC LIMIT 1",
            (rule_id,),
        ))


def recent_evaluations(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        return _rows(conn.execute(
            "SELECT e.*, r.raw_instruction, s.name AS sensor_name"
            " FROM evaluations e"
            " JOIN rules r ON r.id = e.rule_id"
            " JOIN sensors s ON s.id = r.sensor_id"
            " ORDER BY e.id DESC LIMIT ?",
            (limit,),
        ))


# ---------- alerts ----------

def create_alert(rule_id: int, evaluation_id: int, message: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO alerts(rule_id, evaluation_id, message) VALUES (?,?,?)",
            (rule_id, evaluation_id, message),
        )
        return cur.lastrowid


def last_alert_for_rule(rule_id: int) -> Optional[dict]:
    with get_conn() as conn:
        return _row(conn.execute(
            "SELECT * FROM alerts WHERE rule_id = ? ORDER BY id DESC LIMIT 1",
            (rule_id,),
        ))


def list_alerts(unacked_only: bool = False) -> list[dict]:
    with get_conn() as conn:
        q = ("SELECT a.*, s.name AS sensor_name FROM alerts a"
             " JOIN rules r ON r.id = a.rule_id"
             " JOIN sensors s ON s.id = r.sensor_id")
        if unacked_only:
            q += " WHERE a.acknowledged = 0"
        q += " ORDER BY a.id DESC"
        return _rows(conn.execute(q))


def ack_alert(alert_id: int) -> bool:
    """Returns True if an alert row was actually acknowledged."""
    with get_conn() as conn:
        cur = conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?",
                           (alert_id,))
        return cur.rowcount > 0
