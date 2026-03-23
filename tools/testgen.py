"""
testgen.py — Test generation tool for the AI code assistant.

Aligns with orchestrator patterns:
  - Typed TestgenResult return
  - Three-pass JSON parser
  - Per-test-file validation
  - Passes Role.TESTGEN so the LLM registry selects the right model
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from orchestrator.prompt_templates import testgen_system_prompt
from tools.llm_client import LLMResponse, Role, call_llm

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_MAX_TEST_FILES = 10


# ─────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────

@dataclass
class TestFile:
    path: str
    content: str

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.path:
            errors.append("Test file has an empty path.")
        if not self.path.startswith("tests/"):
            errors.append(f"Test file {self.path!r} is not under tests/.")
        if ".." in self.path or self.path.startswith("/"):
            errors.append(f"Unsafe test path rejected: {self.path!r}")
        if not self.content.strip():
            errors.append(f"Test file {self.path!r} has empty content.")
        if "def test_" not in self.content:
            errors.append(f"Test file {self.path!r} contains no test functions (def test_*).")
        return errors


@dataclass
class TestgenResult:
    tests: list[TestFile]
    llm_response: LLMResponse | None = None
    validation_warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        return bool(self.tests) and self.error is None

    def file_paths(self) -> list[str]:
        return [t.path for t in self.tests]

    def to_dict(self) -> dict:
        return {
            "tests": [{"path": t.path, "content": t.content} for t in self.tests]
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
            if isinstance(parsed, dict) and "tests" in parsed:
                return parsed["tests"]
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError(
        f"Could not extract 'tests' array from LLM response.\n"
        f"Raw (first 400 chars):\n{raw[:400]}"
    )


def _build_test_files(raw_tests: list) -> tuple[list[TestFile], list[str]]:
    tests: list[TestFile] = []
    warnings: list[str] = []

    if len(raw_tests) > _MAX_TEST_FILES:
        warnings.append(f"LLM returned {len(raw_tests)} test files; capping at {_MAX_TEST_FILES}.")
        raw_tests = raw_tests[:_MAX_TEST_FILES]

    for i, raw in enumerate(raw_tests):
        if not isinstance(raw, dict):
            warnings.append(f"Test[{i}] is not a dict — skipped.")
            continue
        tf = TestFile(path=raw.get("path", ""), content=raw.get("content", ""))
        errs = tf.validate()
        if any("Unsafe" in e or "empty path" in e for e in errs):
            logger.warning("Dropping test file[%d]: %s", i, errs)
            warnings.extend(errs)
            continue
        if errs:
            warnings.extend(errs)
        tests.append(tf)

    return tests, warnings


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def generate_tests(
    spec: str,
    code_snapshot: str | dict[str, str],
    *,
    max_tokens: int = 4096,
) -> TestgenResult:
    """
    Generate pytest test files for a given spec and code snapshot.

    Args:
        spec:           Natural-language description of the project requirements.
        code_snapshot:  Either a raw string of code, or a {path: content} dict
                        (as returned by CodegenResult.source_snapshot()).
        max_tokens:     Completion token budget.

    Returns:
        TestgenResult with .tests, .success, .to_dict(), etc.
    """
    if not spec or not spec.strip():
        raise ValueError("spec must not be empty.")

    # Normalise code_snapshot to a formatted string
    if isinstance(code_snapshot, dict):
        snapshot_str = "\n\n".join(
            f"### {path}\n```python\n{content}\n```"
            for path, content in code_snapshot.items()
        )
    else:
        snapshot_str = code_snapshot

    user_prompt = (
        f"Spec:\n{spec.strip()}\n\n"
        f"Code Snapshot:\n{snapshot_str}"
    )

    llm_resp = call_llm(
        system_prompt=testgen_system_prompt,
        user_prompt=user_prompt,
        role=Role.TESTGEN,
        json_mode=True,
        max_tokens=max_tokens,
    )

    if not llm_resp.success:
        logger.error("Testgen LLM call failed: %s", llm_resp.error)
        return TestgenResult(tests=[], llm_response=llm_resp, error=llm_resp.error)

    try:
        raw_tests = _parse_response(llm_resp.content)
        tests, warnings = _build_test_files(raw_tests)

        if warnings:
            logger.warning("Testgen warnings: %s", warnings)

        if not tests:
            return TestgenResult(
                tests=[],
                llm_response=llm_resp,
                validation_warnings=warnings,
                error="No valid test files produced.",
            )

        logger.info("Testgen produced %d test file(s): %s", len(tests), [t.path for t in tests])
        return TestgenResult(
            tests=tests,
            llm_response=llm_resp,
            validation_warnings=warnings,
        )

    except ValueError as exc:
        logger.error("Testgen parse error: %s", exc)
        return TestgenResult(tests=[], llm_response=llm_resp, error=str(exc))