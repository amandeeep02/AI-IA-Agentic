"""
codegen.py — Code generation tool for the AI code assistant.

Aligns with orchestrator patterns:
  - Returns a typed CodegenResult (not a bare dict)
  - Three-pass JSON parser matching planner/critic style
  - Per-file validation with path traversal and size guards
  - Passes the correct LLM Role so the stronger model is selected
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from orchestrator.prompt_templates import codegen_system_prompt
from tools.llm_client import LLMResponse, Role, call_llm

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_MAX_FILE_LINES = 400     # hard ceiling — warn if generated file exceeds this
_MAX_FILES      = 20      # sanity cap on total files returned


# ─────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────

@dataclass
class GeneratedFile:
    path: str
    content: str

    @property
    def line_count(self) -> int:
        return self.content.count("\n") + 1

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.path:
            errors.append("File has an empty path.")
        if ".." in self.path or self.path.startswith("/"):
            errors.append(f"Unsafe path rejected: {self.path!r}")
        if not self.content.strip():
            errors.append(f"File {self.path!r} has empty content.")
        if self.line_count > _MAX_FILE_LINES:
            errors.append(
                f"File {self.path!r} has {self.line_count} lines "
                f"(limit {_MAX_FILE_LINES}) — may need splitting."
            )
        return errors


@dataclass
class CodegenResult:
    files: list[GeneratedFile]
    llm_response: LLMResponse | None = None
    validation_warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        return bool(self.files) and self.error is None

    def file_paths(self) -> list[str]:
        return [f.path for f in self.files]

    def get_file(self, path: str) -> GeneratedFile | None:
        return next((f for f in self.files if f.path == path), None)

    def source_snapshot(self) -> dict[str, str]:
        """Return {path: content} for all .py files — used by the analyzer."""
        return {f.path: f.content for f in self.files if f.path.endswith(".py")}

    def to_dict(self) -> dict:
        return {
            "files": [{"path": f.path, "content": f.content} for f in self.files]
        }


# ─────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1) if match else text.strip()


def _extract_json_object(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response.")
    return text[start : end + 1]


def _parse_response(raw: str) -> list[dict]:
    import json

    for transform in (_strip_fences, _extract_json_object):
        candidate = transform(raw)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "files" in parsed:
                return parsed["files"]
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError(
        f"Could not extract 'files' array from LLM response.\n"
        f"Raw (first 400 chars):\n{raw[:400]}"
    )


def _build_files(raw_files: list) -> tuple[list[GeneratedFile], list[str]]:
    files: list[GeneratedFile] = []
    warnings: list[str] = []

    if len(raw_files) > _MAX_FILES:
        warnings.append(
            f"LLM returned {len(raw_files)} files; capping at {_MAX_FILES}."
        )
        raw_files = raw_files[:_MAX_FILES]

    for i, raw in enumerate(raw_files):
        if not isinstance(raw, dict):
            warnings.append(f"File[{i}] is not a dict — skipped.")
            continue
        gf = GeneratedFile(
            path=raw.get("path", ""),
            content=raw.get("content", ""),
        )
        errs = gf.validate()
        if any("Unsafe path" in e or "empty path" in e for e in errs):
            logger.warning("Dropping file[%d]: %s", i, errs)
            warnings.extend(errs)
            continue
        if errs:
            warnings.extend(errs)   # soft warnings (e.g. line count)
        files.append(gf)

    return files, warnings


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def generate_code(user_requirement: str) -> CodegenResult:
    """
    Generate a runnable project from a natural-language requirement.

    Args:
        user_requirement: What to build.

    Returns:
        CodegenResult with .files, .success, .source_snapshot(), etc.
    """
    if not user_requirement or not user_requirement.strip():
        raise ValueError("user_requirement must not be empty.")

    system_prompt = codegen_system_prompt.replace(
        "<USER_REQUIREMENT>", user_requirement.strip()
    )

    llm_resp = call_llm(
        system_prompt=system_prompt,
        user_prompt="Generate the project now.",
        role=Role.CODEGEN,
        json_mode=True,
        max_tokens=6000,
    )

    if not llm_resp.success:
        logger.error("Codegen LLM call failed: %s", llm_resp.error)
        return CodegenResult(files=[], llm_response=llm_resp, error=llm_resp.error)

    try:
        raw_files = _parse_response(llm_resp.content)
        files, warnings = _build_files(raw_files)

        if warnings:
            logger.warning("Codegen warnings: %s", warnings)

        if not files:
            return CodegenResult(
                files=[],
                llm_response=llm_resp,
                validation_warnings=warnings,
                error="No valid files produced.",
            )

        logger.info("Codegen produced %d file(s): %s", len(files), [f.path for f in files])
        return CodegenResult(
            files=files,
            llm_response=llm_resp,
            validation_warnings=warnings,
        )

    except ValueError as exc:
        logger.error("Codegen parse error: %s", exc)
        return CodegenResult(
            files=[],
            llm_response=llm_resp,
            error=str(exc),
        )