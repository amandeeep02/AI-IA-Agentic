"""
logger.py — Structured JSONL run logger.

Features:
  - Thread-safe file writes via a lock
  - ISO-8601 timestamps
  - Log level field (info | warning | error)
  - Log rotation when a single run file exceeds MAX_FILE_BYTES
  - In-memory tail for fast recent-event queries
  - read_run() helper to replay a full run log
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_stdlib_logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 50 * 1024 * 1024   # 50 MB before rotation
_TAIL_SIZE = 200                      # in-memory ring buffer per run


class Logger:
    """
    Append-only structured logger that writes one JSONL file per run_id.

    Usage:
        log = Logger()
        log.info(run_id, "plan", {"tasks": [...]})
        log.warning(run_id, "patch", {"msg": "patch path suspicious"})
        recent = log.tail(run_id, n=10)
    """

    def __init__(self, log_dir: str = "report/logs") -> None:
        self._log_dir = Path(log_dir)
        self._lock = threading.Lock()
        # ring buffer: run_id → deque of entry dicts
        self._tail_cache: dict[str, deque[dict[str, Any]]] = {}

    # ── Public write API ────────────────────────────────────────

    def info(self, run_id: str, stage: str, detail: Any = None) -> None:
        self._write(run_id, stage, detail, level="info")

    def warning(self, run_id: str, stage: str, detail: Any = None) -> None:
        self._write(run_id, stage, detail, level="warning")

    def error(self, run_id: str, stage: str, detail: Any = None) -> None:
        self._write(run_id, stage, detail, level="error")

    # Legacy alias so old callers still work without changes
    def log(self, run_id: str, stage: str, detail: Any = None) -> None:
        self.info(run_id, stage, detail)

    # ── Public read API ─────────────────────────────────────────

    def tail(self, run_id: str, n: int = 20) -> list[dict[str, Any]]:
        """Return the last `n` log entries for this run (in-memory)."""
        buf = self._tail_cache.get(run_id)
        if buf is None:
            return []
        items = list(buf)
        return items[-n:]

    def read_run(self, run_id: str) -> list[dict[str, Any]]:
        """Read and parse the full JSONL file for a run_id from disk."""
        path = self._run_path(run_id)
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        _stdlib_logger.warning("Corrupt log line in %s: %r", path, line)
        return entries

    # ── Internals ───────────────────────────────────────────────

    def _run_path(self, run_id: str) -> Path:
        return self._log_dir / f"{run_id}.jsonl"

    def _rotated_path(self, run_id: str) -> Path:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
        return self._log_dir / f"{run_id}.{ts}.jsonl"

    def _write(self, run_id: str, stage: str, detail: Any, level: str) -> None:
        entry: dict[str, Any] = {
            "time": datetime.now(tz=timezone.utc).isoformat(),
            "run_id": run_id,
            "level": level,
            "stage": stage,
            "detail": detail,
        }
        line = json.dumps(entry, default=str) + "\n"

        with self._lock:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            path = self._run_path(run_id)

            # Rotate if file is too large
            if path.exists() and path.stat().st_size >= _MAX_FILE_BYTES:
                path.rename(self._rotated_path(run_id))
                _stdlib_logger.info("Rotated log for run %s", run_id)

            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)

        # Update in-memory tail
        if run_id not in self._tail_cache:
            self._tail_cache[run_id] = deque(maxlen=_TAIL_SIZE)
        self._tail_cache[run_id].append(entry)

        # Mirror to stdlib logger for terminal visibility
        _stdlib_logger.log(
            {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}.get(
                level, logging.DEBUG
            ),
            "[%s] %s — %s",
            run_id,
            stage,
            detail,
        )