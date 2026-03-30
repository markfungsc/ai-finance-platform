"""Central logging setup for CLI scripts and the API."""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | None = None) -> None:
    """
    Configure the root logger once (idempotent).

    Safe under uvicorn: if the root logger already has handlers, only the level is updated.
    For plain ``python script.py``, installs a stderr StreamHandler with a consistent format.

    Level: ``level`` argument, else ``LOG_LEVEL`` env (default ``INFO``).
    """
    root = logging.getLogger()
    lvl_name = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    root.setLevel(getattr(logging, lvl_name, logging.INFO))

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
        )
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; ensures ``configure_logging`` has run at least once."""
    configure_logging()
    return logging.getLogger(name)
