"""Central configuration. Every value is env-overridable.

Tests monkeypatch these module attributes directly (e.g. config.DB_PATH),
so all other modules must read them at *call time* via `config.X`, never
`from config import X` at import time.
"""
import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("SENTINEL_BASE_DIR", Path(__file__).resolve().parent.parent))

DB_PATH = os.environ.get("SENTINEL_DB", str(BASE_DIR / "sentinel.db"))

# llama-server (OpenAI-compatible) endpoint
MODEL_ENDPOINT = os.environ.get("SENTINEL_MODEL_ENDPOINT", "http://localhost:8080")
MODEL_NAME = os.environ.get("SENTINEL_MODEL_NAME", "local")  # llama-server ignores it
LLM_TIMEOUT_S = float(os.environ.get("SENTINEL_LLM_TIMEOUT", "180"))

# Scheduler
POLL_SECONDS = int(os.environ.get("SENTINEL_POLL_SECONDS", "300"))

# Ingestion directories
INBOX_DIR = os.environ.get("SENTINEL_INBOX", str(BASE_DIR / "inbox"))
IMAGES_DIR = os.environ.get("SENTINEL_IMAGES", str(BASE_DIR / "data" / "images"))

# Web server
HOST = os.environ.get("SENTINEL_HOST", "127.0.0.1")
PORT = int(os.environ.get("SENTINEL_PORT", "8000"))

# Rule defaults
DEFAULT_COOLDOWN_MINUTES = 240
FUZZY_SENSOR_CUTOFF = 0.6
