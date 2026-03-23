"""
memory.py — Persistent run and patch memory backed by SQLite.

Schema:
  runs          — one row per controller run
  patches       — one row per applied patch
  bug_patterns  — aggregated error signatures for future diagnostics

All public methods are thread-safe via a reentrant lock.
Connection is opened once per instance (check_same_thread=False).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    requirement     TEXT NOT NULL,
    start_time      TEXT NOT NULL,
    end_time        TEXT,
    success         INTEGER NOT NULL DEFAULT 0,
    iterations      INTEGER NOT NULL DEFAULT 0,
    task_count      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS patches (
    patch_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES runs(run_id),
    file_path       TEXT NOT NULL,
    patch_time      TEXT NOT NULL,
    iteration       INTEGER NOT NULL DEFAULT 0,
    explanation     TEXT,
    confidence      TEXT,
    summary         TEXT
);

CREATE TABLE IF NOT EXISTS bug_patterns (
    pattern_id      TEXT PRIMARY KEY,
    signature       TEXT NOT NULL UNIQUE,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    occurrences     INTEGER NOT NULL DEFAULT 1,
    example_patch   TEXT
);

CREATE INDEX IF NOT EXISTS idx_patches_run ON patches(run_id);
CREATE INDEX IF NOT EXISTS idx_patterns_sig ON bug_patterns(signature);
"""


# ─────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    run_id: str
    requirement: str
    start_time: str
    end_time: str | None
    success: bool
    iterations: int
    task_count: int


@dataclass
class PatchRecord:
    patch_id: str
    run_id: str
    file_path: str
    patch_time: str
    iteration: int
    explanation: str
    confidence: str
    summary: str


# ─────────────────────────────────────────────────────────────
# Memory
# ─────────────────────────────────────────────────────────────

class Memory:
    """
    Thread-safe SQLite-backed memory store for the AI code assistant.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = "memory.sqlite") -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        logger.info("Memory store opened: %s", db_path)

    # ── Setup ────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        logger.info("Memory store closed: %s", self._db_path)

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Run API ──────────────────────────────────────────────

    def start_run(self, run_id: str, requirement: str, task_count: int = 0) -> None:
        """Insert a run record at the moment execution begins."""
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO runs
                    (run_id, requirement, start_time, success, iterations, task_count)
                VALUES (?, ?, ?, 0, 0, ?)
                """,
                (run_id, requirement, now, task_count),
            )
            self._conn.commit()

    def finish_run(
        self,
        run_id: str,
        success: bool,
        iterations: int,
    ) -> None:
        """Update a run record when it completes."""
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                UPDATE runs
                   SET end_time = ?, success = ?, iterations = ?
                 WHERE run_id = ?
                """,
                (now, int(success), iterations, run_id),
            )
            self._conn.commit()

    # Legacy single-call API — still works
    def log_run(
        self,
        run_id: str,
        requirement: str,
        start_time: str,
        end_time: str,
        success: bool,
        iterations: int = 0,
        task_count: int = 0,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, requirement, start_time, end_time, success, iterations, task_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, requirement, start_time, end_time, int(success), iterations, task_count),
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return RunRecord(
            run_id=row["run_id"],
            requirement=row["requirement"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            success=bool(row["success"]),
            iterations=row["iterations"],
            task_count=row["task_count"],
        )

    def recent_runs(self, limit: int = 20) -> list[RunRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY start_time DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            RunRecord(
                run_id=r["run_id"],
                requirement=r["requirement"],
                start_time=r["start_time"],
                end_time=r["end_time"],
                success=bool(r["success"]),
                iterations=r["iterations"],
                task_count=r["task_count"],
            )
            for r in rows
        ]

    # ── Patch API ────────────────────────────────────────────

    def log_patch(
        self,
        patch_id: str,
        run_id: str,
        file_path: str,
        iteration: int = 0,
        explanation: str = "",
        confidence: str = "",
        summary: str = "",
    ) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO patches
                    (patch_id, run_id, file_path, patch_time, iteration,
                     explanation, confidence, summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (patch_id, run_id, file_path, now, iteration, explanation, confidence, summary),
            )
            self._conn.commit()

    def patches_for_run(self, run_id: str) -> list[PatchRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM patches WHERE run_id = ? ORDER BY patch_time",
                (run_id,),
            ).fetchall()
        return [
            PatchRecord(
                patch_id=r["patch_id"],
                run_id=r["run_id"],
                file_path=r["file_path"],
                patch_time=r["patch_time"],
                iteration=r["iteration"],
                explanation=r["explanation"],
                confidence=r["confidence"],
                summary=r["summary"],
            )
            for r in rows
        ]

    # ── Bug pattern API ──────────────────────────────────────

    def record_bug_pattern(self, signature: str, example_patch: str = "") -> None:
        """
        Upsert a bug pattern signature.
        On first sight: insert. On repeat: increment counter + update last_seen.
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO bug_patterns (pattern_id, signature, first_seen, last_seen,
                                          occurrences, example_patch)
                VALUES (hex(randomblob(8)), ?, ?, ?, 1, ?)
                ON CONFLICT(signature) DO UPDATE SET
                    last_seen   = excluded.last_seen,
                    occurrences = occurrences + 1,
                    example_patch = CASE WHEN excluded.example_patch != ''
                                         THEN excluded.example_patch
                                         ELSE example_patch END
                """,
                (signature, now, now, example_patch),
            )
            self._conn.commit()

    def top_bug_patterns(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT signature, occurrences, first_seen, last_seen, example_patch
                  FROM bug_patterns
                 ORDER BY occurrences DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]