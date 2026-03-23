"""
controller.py — Top-level orchestrator for the AI code assistant.

Responsibilities:
  - Coordinate plan → codegen → test → patch loop
  - Provide clean progress reporting
  - Write all intermediate artifacts safely
  - Record everything in Memory and Logger
  - Return a rich RunSummary instead of a bare tuple
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.critic import AnalysisResult, analyze_failure
from orchestrator.logger import Logger
from orchestrator.memory import Memory
from orchestrator.planner import PlanResult, plan
from tools.codegen import generate_code
from tools.sandbox import Sandbox

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────

@dataclass
class IterationRecord:
    iteration: int
    test_result: dict[str, Any]
    passed: bool
    analysis: AnalysisResult | None = None
    patched_files: list[str] = field(default_factory=list)


@dataclass
class RunSummary:
    run_id: str
    success: bool
    iterations_used: int
    max_iterations: int
    plan: PlanResult | None
    run_dir: str
    iterations: list[IterationRecord] = field(default_factory=list)
    error: str | None = None

    def __str__(self) -> str:
        status = "✓ PASSED" if self.success else "✗ FAILED"
        lines = [
            f"{status}  run_id={self.run_id}",
            f"  dir:        {self.run_dir}",
            f"  iterations: {self.iterations_used}/{self.max_iterations}",
        ]
        if self.plan:
            lines.append(f"  plan tasks: {len(self.plan.tasks)}")
        if self.error:
            lines.append(f"  error:      {self.error}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# File-writing helpers
# ─────────────────────────────────────────────────────────────

def _safe_write(base_dir: Path, rel_path: str, content: str) -> Path:
    """
    Write content to base_dir / rel_path, creating parent dirs as needed.
    Rejects paths that escape the base directory (path traversal guard).
    """
    target = (base_dir / rel_path).resolve()
    if not str(target).startswith(str(base_dir.resolve())):
        raise ValueError(f"Path traversal attempt blocked: {rel_path!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _write_files(run_dir: Path, files: list[dict[str, Any]]) -> list[str]:
    """Write generated files to disk; return list of written relative paths."""
    written: list[str] = []
    for f in files:
        rel = f.get("path", "")
        content = f.get("content", "")
        if not rel:
            logger.warning("Skipping file entry with no path.")
            continue
        try:
            _safe_write(run_dir, rel, content)
            written.append(rel)
        except ValueError as exc:
            logger.error("Blocked write: %s", exc)
    return written


# ─────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────

class Controller:
    """
    Orchestrates the full requirement → working code pipeline.

    Args:
        mode:           Sandbox execution mode ("local" or "docker").
        max_iters:      Maximum patch-and-retry iterations.
        runs_dir:       Root directory for per-run artifacts.
        low_confidence_threshold:
                        If the analyzer returns "low" confidence, skip
                        auto-applying the patch and bail out early.
    """

    def __init__(
        self,
        mode: str = "local",
        max_iters: int = 4,
        runs_dir: str = "examples",
        low_confidence_threshold: bool = True,
    ) -> None:
        self.sandbox = Sandbox(mode=mode)
        self.log = Logger()
        self.memory = Memory()
        self.max_iters = max_iters
        self.runs_dir = Path(runs_dir)
        self.low_confidence_threshold = low_confidence_threshold

    # ── Public API ───────────────────────────────────────────

    def run(self, requirement: str) -> RunSummary:
        """
        Execute the full pipeline for a natural-language requirement.

        Returns:
            RunSummary with outcome, run_id, artifact directory, and iteration log.
        """
        if not requirement or not requirement.strip():
            raise ValueError("requirement must not be empty.")

        run_id = f"run_{uuid.uuid4().hex[:8]}"
        run_dir = self.runs_dir / run_id
        start_time = datetime.now(tz=timezone.utc)

        logger.info("═══ Starting run %s ═══", run_id)
        self.log.info(run_id, "start", {"requirement": requirement})
        self.memory.start_run(run_id, requirement)

        summary = RunSummary(
            run_id=run_id,
            success=False,
            iterations_used=0,
            max_iterations=self.max_iters,
            plan=None,
            run_dir=str(run_dir),
        )

        try:
            # ── 1. Plan ─────────────────────────────────────
            plan_result = plan(requirement, raise_on_failure=False)
            summary.plan = plan_result
            self.log.info(run_id, "plan", {"tasks": [t.to_dict() for t in plan_result.tasks]})

            if not plan_result.success:
                logger.warning("Planner failed or returned no tasks — proceeding with codegen only.")
            else:
                logger.info(plan_result.summary())

            clarifications = plan_result.needs_clarification() if plan_result.success else []
            if clarifications:
                logger.warning(
                    "%d task(s) need clarification: %s",
                    len(clarifications),
                    [t.id for t in clarifications],
                )

            # ── 2. Code generation ───────────────────────────
            logger.info("Generating code…")
            code_json = generate_code(requirement)
            files = code_json.get("files", [])
            if not files:
                raise RuntimeError("Code generator returned no files.")

            written = _write_files(run_dir, files)
            self.log.info(run_id, "write_files", {"dir": str(run_dir), "files": written})
            logger.info("Wrote %d file(s) to %s", len(written), run_dir)

            # ── 3. Build a quick context snapshot for the analyzer ──
            source_snapshot: dict[str, str] = {}
            for rel in written:
                p = run_dir / rel
                if p.suffix == ".py" and p.exists():
                    source_snapshot[rel] = p.read_text(encoding="utf-8")

            # ── 4. Test → patch loop ─────────────────────────
            for iter_num in range(1, self.max_iters + 1):
                logger.info("─── Iteration %d/%d ───", iter_num, self.max_iters)
                summary.iterations_used = iter_num

                test_result = self.sandbox.run_tests(str(run_dir))
                self.log.info(run_id, "test_run", {**test_result, "iteration": iter_num})
                passed = test_result.get("rc", 1) == 0

                record = IterationRecord(
                    iteration=iter_num,
                    test_result=test_result,
                    passed=passed,
                )

                if passed:
                    logger.info("All tests passed ✓")
                    summary.success = True
                    summary.iterations.append(record)
                    break

                if iter_num == self.max_iters:
                    logger.warning("Max iterations reached without passing tests.")
                    summary.iterations.append(record)
                    break

                # ── Analyze failure ──────────────────────────
                logger.info("Tests failed — analyzing…")
                analysis = analyze_failure(
                    test_result.get("stdout", ""),
                    test_result.get("stderr", ""),
                    context_files=source_snapshot,
                )

                if analysis.confidence == "low" and self.low_confidence_threshold:
                    logger.warning(
                        "Analyzer confidence is LOW — skipping auto-patch to avoid churn."
                    )
                    record.analysis = analysis
                    summary.iterations.append(record)
                    summary.error = (
                        f"Stopped at iteration {iter_num}: "
                        f"low-confidence analysis. Root cause: {analysis.root_cause}"
                    )
                    break

                # ── Apply patches ────────────────────────────
                patched: list[str] = []
                for patch in analysis.patches:
                    try:
                        _safe_write(run_dir, patch.path, patch.new_content)
                        patched.append(patch.path)
                        patch_id = uuid.uuid4().hex[:8]
                        self.memory.log_patch(
                            patch_id=patch_id,
                            run_id=run_id,
                            file_path=patch.path,
                            iteration=iter_num,
                            explanation=analysis.explanation,
                            confidence=analysis.confidence,
                            summary=analysis.root_cause,
                        )
                        # Refresh snapshot so next iteration has updated source
                        if patch.path in source_snapshot:
                            source_snapshot[patch.path] = patch.new_content
                    except ValueError as exc:
                        logger.error("Blocked patch write: %s", exc)

                self.log.info(
                    run_id,
                    "apply_patch",
                    {
                        "iteration": iter_num,
                        "patched_files": patched,
                        "root_cause": analysis.root_cause,
                        "explanation": analysis.explanation,
                        "confidence": analysis.confidence,
                    },
                )

                if analysis.root_cause:
                    self.memory.record_bug_pattern(
                        signature=analysis.root_cause[:200],
                        example_patch=patched[0] if patched else "",
                    )

                record.analysis = analysis
                record.patched_files = patched
                summary.iterations.append(record)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Controller run %s raised unexpectedly", run_id)
            summary.error = str(exc)
            self.log.error(run_id, "fatal_error", {"error": str(exc)})

        finally:
            self.memory.finish_run(
                run_id=run_id,
                success=summary.success,
                iterations=summary.iterations_used,
            )
            elapsed = (datetime.now(tz=timezone.utc) - start_time).total_seconds()
            self.log.info(
                run_id,
                "finish",
                {"success": summary.success, "elapsed_seconds": round(elapsed, 2)},
            )
            logger.info("═══ Run %s finished in %.1fs. Success=%s ═══", run_id, elapsed, summary.success)

        return summary


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Code Assistant — Controller")
    parser.add_argument("requirement", help="Natural-language requirement to implement")
    parser.add_argument("--mode", default="local", choices=["local", "docker"], help="Sandbox mode")
    parser.add_argument("--max-iters", type=int, default=4, help="Max patch iterations")
    parser.add_argument("--runs-dir", default="examples", help="Root directory for run artifacts")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    parser.add_argument(
        "--no-confidence-gate",
        action="store_true",
        help="Apply patches even when analyzer confidence is low",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    ctrl = Controller(
        mode=args.mode,
        max_iters=args.max_iters,
        runs_dir=args.runs_dir,
        low_confidence_threshold=not args.no_confidence_gate,
    )
    result = ctrl.run(args.requirement)
    print(result)