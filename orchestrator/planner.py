"""
planner.py — Requirement decomposition layer for the AI code assistant.

Responsibilities:
  - Call the LLM planner with structured retry/backoff logic
  - Robustly parse and validate the returned task array
  - Enrich tasks with runtime metadata before returning
  - Provide a typed public interface via PlanTask / PlanResult
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from orchestrator.prompt_templates import planner_system_prompt
from tools.llm_client import call_llm

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────

@dataclass
class PlanTask:
    id: str
    title: str
    description: str
    est_hours: float
    depends_on: list[str] = field(default_factory=list)
    test_hint: str = ""
    clarification_needed: str | None = None

    # Populated at runtime — not from LLM
    status: str = "pending"   # pending | in_progress | done | failed

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlanTask":
        return cls(
            id=raw["id"],
            title=raw["title"],
            description=raw["description"],
            est_hours=float(raw.get("est_hours", 1.0)),
            depends_on=raw.get("depends_on") or [],
            test_hint=raw.get("test_hint", ""),
            clarification_needed=raw.get("clarification_needed"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "est_hours": self.est_hours,
            "depends_on": self.depends_on,
            "test_hint": self.test_hint,
            "clarification_needed": self.clarification_needed,
            "status": self.status,
        }


@dataclass
class PlanResult:
    tasks: list[PlanTask]
    raw_response: str
    attempts: int
    error: str | None = None

    @property
    def success(self) -> bool:
        return bool(self.tasks) and self.error is None

    @property
    def total_hours(self) -> float:
        return sum(t.est_hours for t in self.tasks)

    def needs_clarification(self) -> list[PlanTask]:
        return [t for t in self.tasks if t.clarification_needed]

    def execution_order(self) -> list[PlanTask]:
        """Topological sort — returns tasks in safe execution order."""
        visited: set[str] = set()
        order: list[PlanTask] = []
        by_id = {t.id: t for t in self.tasks}

        def visit(task: PlanTask) -> None:
            if task.id in visited:
                return
            for dep_id in task.depends_on:
                if dep_id in by_id:
                    visit(by_id[dep_id])
            visited.add(task.id)
            order.append(task)

        for task in self.tasks:
            visit(task)
        return order

    def summary(self) -> str:
        lines = [f"Plan: {len(self.tasks)} tasks  |  ~{self.total_hours:.1f} h total"]
        for task in self.execution_order():
            flag = " ⚠ clarification needed" if task.clarification_needed else ""
            deps = f" (after {', '.join(task.depends_on)})" if task.depends_on else ""
            lines.append(f"  [{task.id}] {task.title}  ({task.est_hours}h){deps}{flag}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text.strip()


def _extract_json_array(text: str) -> str:
    """
    Best-effort: find the outermost [...] in the response,
    even if the model prefixed it with prose.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in LLM response.")
    return text[start : end + 1]


def _parse_raw_response(raw: str) -> list[dict[str, Any]]:
    """
    Three-pass parser:
      1. Strip fences → parse
      2. Extract outermost array → parse
      3. Raise with a helpful message
    """
    for transform in (_strip_fences, _extract_json_array):
        candidate = transform(raw)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
            raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}.")
        except json.JSONDecodeError:
            continue

    raise ValueError(
        "Could not parse a JSON array from the LLM response.\n"
        f"Raw response (first 400 chars):\n{raw[:400]}"
    )


# ─────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────

_REQUIRED_TASK_KEYS = {"id", "title", "description"}


def _validate_task(raw: dict[str, Any], index: int) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: list[str] = []
    missing = _REQUIRED_TASK_KEYS - raw.keys()
    if missing:
        errors.append(f"Task[{index}] missing required keys: {missing}")

    if "est_hours" in raw:
        try:
            h = float(raw["est_hours"])
            if not (0 < h <= 8):
                errors.append(f"Task[{index}] est_hours={h} out of range (0, 8]")
        except (TypeError, ValueError):
            errors.append(f"Task[{index}] est_hours is not numeric: {raw['est_hours']!r}")

    if "depends_on" in raw and not isinstance(raw["depends_on"], list):
        errors.append(f"Task[{index}] depends_on must be a list")

    return errors


def _build_tasks(raw_list: list[dict[str, Any]]) -> tuple[list[PlanTask], list[str]]:
    """Convert raw dicts → PlanTask objects, collecting all validation errors."""
    tasks: list[PlanTask] = []
    all_errors: list[str] = []

    for i, raw in enumerate(raw_list):
        errs = _validate_task(raw, i)
        if errs:
            all_errors.extend(errs)
            logger.warning("Skipping malformed task %d: %s", i, errs)
            continue
        tasks.append(PlanTask.from_dict(raw))

    return tasks, all_errors


# ─────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────

_RETRY_DELAYS = (0, 2, 5)   # seconds before each attempt (first is immediate)


def plan(
    user_requirement: str,
    *,
    max_retries: int = 3,
    raise_on_failure: bool = False,
) -> PlanResult:
    """
    Decompose a user requirement into an ordered list of PlanTask objects.

    Args:
        user_requirement:  Free-text description of what to build.
        max_retries:       How many LLM attempts before giving up.
        raise_on_failure:  If True, raises RuntimeError instead of returning
                           an empty PlanResult on total failure.

    Returns:
        PlanResult with .tasks, .success, .summary(), etc.
    """
    if not user_requirement or not user_requirement.strip():
        raise ValueError("user_requirement must not be empty.")

    prompt = f"Requirement:\n{user_requirement.strip()}"
    last_error: str = ""
    raw_response: str = ""

    for attempt in range(1, max_retries + 1):
        delay = _RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)]
        if delay:
            logger.info("Planner retry %d/%d — waiting %ds…", attempt, max_retries, delay)
            time.sleep(delay)

        try:
            logger.info("Planner attempt %d/%d", attempt, max_retries)
            raw_response = call_llm(planner_system_prompt, prompt, json_mode=False)
            raw_list = _parse_raw_response(raw_response)
            tasks, validation_errors = _build_tasks(raw_list)

            if validation_errors:
                logger.warning(
                    "Planner returned %d tasks but %d had validation errors.",
                    len(raw_list),
                    len(validation_errors),
                )

            if tasks:
                logger.info(
                    "Plan produced: %d tasks, %.1f h total", len(tasks), sum(t.est_hours for t in tasks)
                )
                return PlanResult(tasks=tasks, raw_response=raw_response, attempts=attempt)

            last_error = "Parsed successfully but zero valid tasks were produced."
            logger.warning("Attempt %d: %s", attempt, last_error)

        except ValueError as exc:
            last_error = str(exc)
            logger.warning("Attempt %d parse error: %s", attempt, last_error)
        except Exception as exc:                          # noqa: BLE001
            last_error = f"Unexpected error: {exc}"
            logger.exception("Attempt %d unexpected failure", attempt)

    # All attempts exhausted
    logger.error("Planner failed after %d attempts. Last error: %s", max_retries, last_error)
    result = PlanResult(tasks=[], raw_response=raw_response, attempts=max_retries, error=last_error)
    if raise_on_failure:
        raise RuntimeError(f"Planner failed: {last_error}")
    return result