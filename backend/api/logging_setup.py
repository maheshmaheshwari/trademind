"""
TradeMind AI — Centralised Logging Setup

Creates a new log file per calendar day: logs/YYYY-MM-DD.log
Rotates automatically at midnight even if the server runs overnight.

Works whether the server is started via:
  • python main.py server
  • uvicorn api.server:app --host 0.0.0.0 --port 8000 [--reload]

How uvicorn interacts with logging
───────────────────────────────────
Uvicorn calls logging.config.dictConfig() which REPLACES root-logger handlers.
To survive that reset we:
  1. Call setup_logging() at module-import level in server.py  (catches the
     case where uvicorn imports the app AFTER configuring its own logging).
  2. Call setup_logging() again inside the FastAPI startup event  (catches the
     case where uvicorn's dictConfig() runs after the import and wipes our
     handlers — the startup event always fires after dictConfig()).
  3. Pass a custom log_config dict to uvicorn.run() (main.py) that tells
     uvicorn to leave the root logger alone, so the two configs coexist.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


# ── Custom handler: rotates to a new YYYY-MM-DD.log file at midnight ──────────

class DailyFileHandler(logging.FileHandler):
    """
    Writes to logs/YYYY-MM-DD.log.
    On every emit() it checks today's date; when midnight passes it
    closes the old file and opens a fresh one with the new date.
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._current_date = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(log_dir, f"{self._current_date}.log")
        super().__init__(filepath, mode="a", encoding="utf-8", delay=False)

    def emit(self, record: logging.LogRecord) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self._current_date = today
            self.close()
            self.baseFilename = os.path.abspath(
                os.path.join(self.log_dir, f"{today}.log")
            )
            self.stream = self._open()
        super().emit(record)


# ── Log format used by both handlers ──────────────────────────────────────────

_FMT = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _has_daily_handler(root: logging.Logger) -> bool:
    """Return True if root already has a DailyFileHandler attached."""
    return any(isinstance(h, DailyFileHandler) for h in root.handlers)


# ── Public API ─────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """
    Attach a DailyFileHandler (DEBUG+) and a StreamHandler (INFO+) to the
    root logger.  Idempotent — calling it multiple times is safe; a second
    DailyFileHandler is never added.
    """
    root = logging.getLogger()

    if _has_daily_handler(root):
        return  # already set up in this process — nothing to do

    log_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(logging.DEBUG)

    # ── File handler (date-rotating) ──────────────────────────────────────────
    file_handler = DailyFileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FMT)
    root.addHandler(file_handler)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(_FMT)
    root.addHandler(console_handler)

    # Quieten noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore", "hpack",
                  "apscheduler", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    today = datetime.now().strftime("%Y-%m-%d")
    logging.getLogger(__name__).info(
        "Logging initialised — writing to %s/%s.log", log_dir, today
    )


def get_uvicorn_log_config(level: str = "INFO") -> dict:
    """
    Return a log_config dict for uvicorn.run() that:
      - keeps uvicorn's own loggers (uvicorn, uvicorn.error) working
      - does NOT reconfigure the root logger (so our DailyFileHandler survives)
      - disables uvicorn's built-in access log (we log requests in middleware)
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,   # ← critical: leaves root alone
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": None,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": level.upper(),
                "propagate": False,
            },
            "uvicorn.error": {
                "level": level.upper(),
                "propagate": True,   # let uvicorn errors reach root (→ file)
            },
            "uvicorn.access": {
                "handlers": [],      # silenced — middleware handles access log
                "level": "WARNING",
                "propagate": False,
            },
        },
    }
