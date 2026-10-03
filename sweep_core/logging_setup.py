import logging
import sys

from sweep_core.config import Settings


def setup_logging(settings: Settings) -> logging.Logger:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    return logging.getLogger("sweep")
