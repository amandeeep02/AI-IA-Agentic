"""
static_analysis.py — Pre-execution safety and quality analysis for generated code.

Checks:
  - Forbidden imports / calls that must block execution
  - Warning-level patterns that should be reviewed
  - Basic structural quality signals (bare except, TODO density, function length)
  - Optional AST-based validation for .py files
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Rule definitions
# ─────────────────────────────────────────────────────────────

# (pattern, human-readable label)
_FORBIDDEN: list[tuple[str, str]] = [
    (r"\bimport\s+socket\b",              "network: import socket"),
    (r"\bimport\s+requests\b",            "network: import requests"),
    (r"\bimport\s+httpx\b",               "network: import httpx"),
    (r"\bos\.system\s*\(",                "shell: os.system()"),
    (r"\bsubprocess\.Popen\s*\(",         "shell: subprocess.Popen()"),
    (r"\bsubprocess\.call\s*\(",          "shell: subprocess.call()"),
    (r"\beval\s*\(",                      "code injection: eval()"),
    (r"\bexec\s*\(",                      "code injection: exec()"),
    (r"\b__import__\s*\(",                "code injection: __import__()"),
    (r"open\s*\(\s*['\"]\/dev\/",         "device access: open('/dev/...')"),
    (r"\bimport\s+ctypes\b",              "unsafe: import ctypes"),
    (r"\bimport\s+pickle\b",              "unsafe deserialization: import pickle"),
]

_WARNINGS: list[tuple[str, str]] = [
    (r"\bimport\s+os\b",                  "review: import os (os.system blocked, but verify usage)"),
    (r"\bimport\s+subprocess\b",          "review: import subprocess (Popen/call blocked)"),
    (r"\bexcept\s*:",                     "quality: bare except clause"),
    (r"#\s*TODO",                         "quality: TODO comment left in generated code"),
    (r"#\s*FIXME",                        "quality: FIXME comment left in generated code"),
    (r"password\s*=\s*['\"][^'\"]+['\"]", "security: hardcoded password literal"),
    (r"secret\s*=\s*['\"][^'\"]+['\"]",   "security: hardcoded secret literal"),
]

_COMPILED_FORBIDDEN = [(re.compile(p), label) for p, label in _FORBIDDEN]
_COMPILED_WARNINGS  = [(re.compile(p), label) for p, label in _WARNINGS]


# ─────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────

@dataclass
class FileAnalysis:
    path: str
    forbidden: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    syntax_errors: list[str] = field(default_factory=list)
    long_functions: list[str] = field(default_factory=list)   # func names > threshold

    @property
    def is_safe(self) -> bool:
        return not self.forbidden and not self.syntax_errors

    @property
    def has_issues(self) -> bool:
        return bool(self.forbidden or self.warnings or self.syntax_errors or self.long_functions)

    def summary(self) -> str:
        parts: list[str] = []
        if self.forbidden:
            parts.append(f"BLOCKED({len(self.forbidden)})")
        if self.warnings:
            parts.append(f"warn({len(self.warnings)})")
        if self.syntax_errors:
            parts.append(f"syntax_err({len(self.syntax_errors)})")
        if self.long_functions:
            parts.append(f"long_fns({len(self.long_functions)})")
        return ", ".join(parts) if parts else "ok"


@dataclass
class ProjectAnalysis:
    files: list[FileAnalysis] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return all(f.is_safe for f in self.files)

    @property
    def blocked_files(self) -> list[FileAnalysis]:
        return [f for f in self.files if f.forbidden or f.syntax_errors]

    @property
    def warned_files(self) -> list[FileAnalysis]:
        return [f for f in self.files if f.warnings or f.long_functions]

    def summary(self) -> str:
        lines = [f"Static analysis: {len(self.files)} file(s) checked"]
        for fa in self.files:
            lines.append(f"  {fa.path}: {fa.summary()}")
        if self.is_safe:
            lines.append("  ✓ All files passed safety checks.")
        else:
            lines.append(f"  ✗ {len(self.blocked_files)} file(s) BLOCKED.")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Core analysis functions
# ─────────────────────────────────────────────────────────────

def _check_content(content: str) -> tuple[list[str], list[str]]:
    """Regex scan — returns (forbidden_labels, warning_labels)."""
    forbidden: list[str] = []
    warnings: list[str] = []
    for pattern, label in _COMPILED_FORBIDDEN:
        if pattern.search(content):
            forbidden.append(label)
    for pattern, label in _COMPILED_WARNINGS:
        if pattern.search(content):
            warnings.append(label)
    return forbidden, warnings


def _check_ast(content: str, path: str, max_func_lines: int = 40) -> tuple[list[str], list[str]]:
    """
    AST-level checks for Python files only.
    Returns (syntax_errors, long_function_names).
    """
    syntax_errors: list[str] = []
    long_functions: list[str] = []
    try:
        tree = ast.parse(content, filename=path)
    except SyntaxError as exc:
        syntax_errors.append(f"SyntaxError at line {exc.lineno}: {exc.msg}")
        return syntax_errors, long_functions

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            length = end - node.lineno
            if length > max_func_lines:
                long_functions.append(
                    f"{node.name}() — {length} lines (limit {max_func_lines})"
                )
    return syntax_errors, long_functions


# ─────────────────────────────────────────────────────────────
# Public API — file level
# ─────────────────────────────────────────────────────────────

def check_file(file_path: str, max_func_lines: int = 40) -> FileAnalysis:
    """
    Analyse a single file on disk.

    Args:
        file_path:      Absolute or relative path to the file.
        max_func_lines: Functions longer than this emit a long_functions warning.

    Returns:
        FileAnalysis — always returns, never raises.
    """
    path = Path(file_path)
    analysis = FileAnalysis(path=str(file_path))

    if not path.exists():
        analysis.syntax_errors.append("File not found.")
        return analysis

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        analysis.syntax_errors.append(f"Could not read file: {exc}")
        return analysis

    forbidden, warnings = _check_content(content)
    analysis.forbidden = forbidden
    analysis.warnings = warnings

    if path.suffix == ".py":
        syntax_errors, long_functions = _check_ast(content, str(path), max_func_lines)
        analysis.syntax_errors = syntax_errors
        analysis.long_functions = long_functions

    if analysis.forbidden:
        logger.warning("BLOCKED %s: %s", file_path, analysis.forbidden)
    elif analysis.warnings:
        logger.info("Warnings in %s: %s", file_path, analysis.warnings)

    return analysis


def check_content(content: str, path: str = "<string>", max_func_lines: int = 40) -> FileAnalysis:
    """
    Analyse in-memory content (e.g. directly from CodegenResult).
    Same as check_file but accepts a string instead of reading from disk.
    """
    analysis = FileAnalysis(path=path)
    forbidden, warnings = _check_content(content)
    analysis.forbidden = forbidden
    analysis.warnings = warnings

    if path.endswith(".py"):
        syntax_errors, long_functions = _check_ast(content, path, max_func_lines)
        analysis.syntax_errors = syntax_errors
        analysis.long_functions = long_functions

    return analysis


# ─────────────────────────────────────────────────────────────
# Public API — project level
# ─────────────────────────────────────────────────────────────

def check_project(project_path: str, extensions: tuple[str, ...] = (".py",)) -> ProjectAnalysis:
    """
    Recursively analyse all matching files in a project directory.

    Args:
        project_path: Root directory to scan.
        extensions:   File extensions to include.

    Returns:
        ProjectAnalysis summarising the entire project.
    """
    root = Path(project_path)
    project_analysis = ProjectAnalysis()

    for file_path in sorted(root.rglob("*")):
        if file_path.suffix in extensions and file_path.is_file():
            # Skip test files and __pycache__
            if "__pycache__" in file_path.parts:
                continue
            fa = check_file(str(file_path))
            project_analysis.files.append(fa)

    logger.info(project_analysis.summary())
    return project_analysis


def check_generated_files(source_snapshot: dict[str, str]) -> ProjectAnalysis:
    """
    Analyse in-memory generated files from CodegenResult.source_snapshot().
    Use this before writing files to disk.

    Args:
        source_snapshot: {relative_path: file_content}

    Returns:
        ProjectAnalysis
    """
    project_analysis = ProjectAnalysis()
    for path, content in source_snapshot.items():
        fa = check_content(content, path=path)
        project_analysis.files.append(fa)
    logger.info(project_analysis.summary())
    return project_analysis