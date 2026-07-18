"""Seed demo data: two sensors and three synthetic sample images dropped into
inbox/breakroom_cam/.

Stdlib-only (hand-rolled PNG encoder) so the demo needs no extra deps.
Run AFTER starting the app once (or it will create the DB itself):
    python scripts/seed_demo.py
"""
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel import config, db  # noqa: E402

W, H = 320, 240


def blank(rgb=(24, 26, 32)):
    return [[list(rgb) for _ in range(W)] for _ in range(H)]


def rect(img, x0, y0, x1, y1, rgb):
    for y in range(max(0, y0), min(H, y1)):
        for x in range(max(0, x0), min(W, x1)):
            img[y][x] = list(rgb)


def png_bytes(img) -> bytes:
    raw = b"".join(b"\x00" + bytes(v for px in row for v in px) for row in img)

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def draw_box(cans: int):
    """A cardboard box seen from above with 0..12 red cans in it."""
    img = blank()
    rect(img, 40, 30, 280, 210, (139, 105, 62))    # box
    rect(img, 55, 45, 265, 195, (92, 68, 40))      # inside
    positions = [(x, y) for y in range(3) for x in range(4)]
    for i, (cx, cy) in enumerate(positions[:cans]):
        x0 = 65 + cx * 50
        y0 = 55 + cy * 48
        rect(img, x0, y0, x0 + 40, y0 + 38, (200, 30, 40))   # can top
        rect(img, x0 + 14, y0 + 12, x0 + 26, y0 + 26, (220, 220, 220))  # tab
    return img


def main() -> None:
    db.init_db()

    for name, kind, location, context in (
        ("breakroom_cam", "image", "2nd floor break room",
         "Fixed camera watching the snack station in the 2nd floor break room. "
         "It shows wire racks and baskets holding snack bags, cookies, fruit "
         "snacks and candy, plus a cardboard box of canned drinks."),
        ("keg_scale", "numeric", "bar", ""),
    ):
        if db.get_sensor_by_name(name) is None:
            db.create_sensor(name, kind, location, context)
            print(f"created sensor {name} ({kind})")
        else:
            print(f"sensor {name} already exists")

    inbox = Path(config.INBOX_DIR) / "breakroom_cam"
    inbox.mkdir(parents=True, exist_ok=True)
    # Numbered so ingestion order (sorted) ends on the EMPTY box — the demo
    # rule evaluates the latest reading, and the latest must be the one that
    # triggers the alert.
    samples = {
        "1_box_full.png": draw_box(12),
        "2_box_half.png": draw_box(5),
        "3_box_empty.png": draw_box(0),
    }
    for fname, img in samples.items():
        path = inbox / fname
        path.write_bytes(png_bytes(img))
        print(f"dropped {path}")

    print("\nDone. Next poll cycle will ingest the images "
          f"(or POST /api/cycle to run one now).")


if __name__ == "__main__":
    main()
