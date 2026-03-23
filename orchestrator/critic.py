"""
critic.py — Failure analysis and patch generation layer.

Responsibilities:
  - Call the analyzer LLM with pytest stdout/stderr
  - Robustly parse and validate the patch response
  - Truncate oversized inputs to stay within context limits
  - Return a typed AnalysisResult
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from orchestrator.prompt_templates import analyzer_system_prompt
from tools.llm_client import call_llm

logger = logging.getLogger(__name__)

# Prevent runaway context usage — trim output beyond this many chars
_MAX_OUTPUT_CHARS = 12_000
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# ─────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────

@dataclass
class FilePatch:
    path: str
    new_content: str

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.path or not self.path.endswith((".py", ".json", ".txt", ".md")):
            errors.append(f"Suspicious patch path: {self.path!r}")
        if not self.new_content.strip():
            errors.append(f"Patch for {self.path!r} has empty content.")
        return errors


@dataclass
class AnalysisResult:
    patches: list[FilePatch]
    root_cause: str
    explanation: str
    confidence: str = "medium"   # high | medium | low
    raw_response: str = ""
    error: str | None = None

    @property
    def success(self) -> bool:
        return bool(self.patches) and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "patches": [{"path": p.path, "new_content": p.new_content} for p in self.patches],
            "root_cause": self.root_cause,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "error": self.error,
        }


_EMPTY_RESULT = AnalysisResult(
    patches=[],
    root_cause="",
    explanation="",
    error="Analysis produced no usable patches.",
)


# ─────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text.strip()


def _extract_json_object(text: str) -> str:
    """Find the outermost {...} even if the model added prose around it."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response.")
    return text[start : end + 1]


def _parse_raw_response(raw: str) -> dict[str, Any]:
    for transform in (_strip_fences, _extract_json_object):
        candidate = transform(raw)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(
        f"Could not parse a JSON object from the LLM response.\n"
        f"Raw (first 400 chars):\n{raw[:400]}"
    )


def _build_patches(raw_patches: list[Any]) -> tuple[list[FilePatch], list[str]]:
    patches: list[FilePatch] = []
    errors: list[str] = []
    for i, raw in enumerate(raw_patches):
        if not isinstance(raw, dict):
            errors.append(f"Patch[{i}] is not a dict.")
            continue
        if "path" not in raw or "new_content" not in raw:
            errors.append(f"Patch[{i}] missing 'path' or 'new_content'.")
            continue
        patch = FilePatch(path=raw["path"], new_content=raw["new_content"])
        patch_errors = patch.validate()
        if patch_errors:
            errors.extend(patch_errors)
            logger.warning("Skipping invalid patch %d: %s", i, patch_errors)
            continue
        patches.append(patch)
    return patches, errors


# ─────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────

def _truncate(text: str, label: str) -> str:
    """Trim to _MAX_OUTPUT_CHARS with a visible marker so the model knows."""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    logger.warning("%s truncated from %d to %d chars", label, len(text), _MAX_OUTPUT_CHARS)
    return text[:_MAX_OUTPUT_CHARS] + f"\n... [{label} TRUNCATED]"


def analyze_failure(
    stdout: str,
    stderr: str,
    *,
    context_files: dict[str, str] | None = None,
) -> AnalysisResult:
    """
    Analyze a pytest failure and return patches that fix it.

    Args:
        stdout:          Captured stdout from the test run.
        stderr:          Captured stderr from the test run.
        context_files:   Optional {path: content} snapshot of relevant source
                         files — gives the LLM more accurate patch targets.

    Returns:
        AnalysisResult with .patches, .explanation, .confidence, etc.
    """
    stdout = _truncate(stdout, "stdout")
    stderr = _truncate(stderr, "stderr")

    prompt_parts = [f"Stdout:\n{stdout}", f"Stderr:\n{stderr}"]
    if context_files:
        snapshot = "\n\n".join(
            f"### {path}\n```python\n{content}\n```"
            for path, content in context_files.items()
        )
        prompt_parts.append(f"Relevant source files:\n{snapshot}")

    prompt = "\n\n".join(prompt_parts)

    raw_response = ""
    try:
        raw_response = call_llm(analyzer_system_prompt, prompt, json_mode=True)
        parsed = _parse_raw_response(raw_response)

        raw_patches = parsed.get("patches", [])
        patches, patch_errors = _build_patches(raw_patches)

        if patch_errors:
            logger.warning("Patch validation issues: %s", patch_errors)

        result = AnalysisResult(
            patches=patches,
            root_cause=parsed.get("root_cause", ""),
            explanation=parsed.get("explanation", ""),
            confidence=parsed.get("confidence", "medium"),
            raw_response=raw_response,
        )

        if not patches:
            result.error = "LLM returned no valid patches."
            logger.warning("Analyzer produced no usable patches.")
        else:
            logger.info(
                "Analysis complete: %d patch(es), confidence=%s",
                len(patches),
                result.confidence,
            )

        return result

    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze_failure raised unexpectedly")
        return AnalysisResult(
            patches=[],
            root_cause="",
            explanation="",
            error=str(exc),
            raw_response=raw_response,
        )