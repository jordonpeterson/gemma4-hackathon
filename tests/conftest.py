import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel import config, db  # noqa: E402


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Isolated SQLite DB + inbox/images dirs per test."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setattr(config, "INBOX_DIR", str(tmp_path / "inbox"))
    monkeypatch.setattr(config, "IMAGES_DIR", str(tmp_path / "images"))
    db.init_db()
    return tmp_path


@pytest.fixture
def demo_sensors(fresh_db):
    cam_id = db.create_sensor("breakroom_cam", "image", "break room")
    keg_id = db.create_sensor("keg_scale", "numeric", "bar")
    door_id = db.create_sensor("door_state", "boolean", "front door")
    return {"breakroom_cam": cam_id, "keg_scale": keg_id, "door_state": door_id}
