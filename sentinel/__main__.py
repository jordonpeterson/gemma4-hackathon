"""Entry point: `python -m sentinel` starts the API + background scheduler."""
import logging

import uvicorn

from sentinel import config, db, scheduler
from sentinel.api import app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    db.init_db()
    scheduler.start()
    try:
        uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="info")
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()
