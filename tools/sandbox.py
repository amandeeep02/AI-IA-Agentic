"""
sandbox.py — Isolated test execution environment.

Supports:
  - Local subprocess execution
  - Docker container execution (network-isolated, resource-limited)
  - Dependency installation with caching
  - Configurable timeouts and result enrichment
  - SandboxResult typed return (aligns with controller.py)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCAL_TIMEOUT_SECS  = 60
_DOCKER_TIMEOUT_SECS = 180
_DOCKER_IMAGE        = "python:3.11-slim"


# ─────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────

@dataclass
class SandboxResult:
    rc: int                          # 0 = pass, >0 = test failure, <0 = infra error
    stdout: str
    stderr: str
    elapsed_seconds: float = 0.0
    mode: str = "local"
    timed_out: bool = False
    infra_error: str | None = None   # set when rc < 0

    @property
    def passed(self) -> bool:
        return self.rc == 0

    @property
    def tests_ran(self) -> bool:
        """True if pytest actually executed (vs infra failure)."""
        return self.rc >= 0

    def to_dict(self) -> dict:
        return {
            "rc": self.rc,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "mode": self.mode,
            "timed_out": self.timed_out,
            "infra_error": self.infra_error,
            "passed": self.passed,
        }


# ─────────────────────────────────────────────────────────────
# Sandbox
# ─────────────────────────────────────────────────────────────

class Sandbox:
    """
    Runs pytest in an isolated environment.

    Args:
        mode:            "local" or "docker".
        local_timeout:   Seconds before a local run is killed.
        docker_timeout:  Seconds before a Docker run is killed.
        docker_image:    Docker image to use for container runs.
        extra_pytest_args: Additional pytest CLI flags (e.g. ["-v", "--tb=short"]).
    """

    def __init__(
        self,
        mode: str = "local",
        local_timeout: int = _LOCAL_TIMEOUT_SECS,
        docker_timeout: int = _DOCKER_TIMEOUT_SECS,
        docker_image: str = _DOCKER_IMAGE,
        extra_pytest_args: list[str] | None = None,
    ) -> None:
        if mode not in ("local", "docker"):
            raise ValueError(f"Unsupported sandbox mode: {mode!r}. Use 'local' or 'docker'.")
        self.mode = mode
        self.local_timeout = local_timeout
        self.docker_timeout = docker_timeout
        self.docker_image = docker_image
        self.extra_pytest_args = extra_pytest_args or []

    # ── Public API ───────────────────────────────────────────

    def run_tests(self, project_path: str) -> SandboxResult:
        """
        Execute pytest in the given project directory.

        Returns:
            SandboxResult — always returns, never raises.
        """
        path = Path(project_path).resolve()
        if not path.exists():
            return SandboxResult(
                rc=-3,
                stdout="",
                stderr="",
                mode=self.mode,
                infra_error=f"Project path does not exist: {path}",
            )

        logger.info("Running tests in %s (mode=%s)", path, self.mode)
        if self.mode == "docker":
            return self._run_docker(path)
        return self._run_local(path)

    # ── Local ────────────────────────────────────────────────

    def _run_local(self, path: Path) -> SandboxResult:
        # Prefer `python -m pytest` so we don't depend on a `pytest` entrypoint
        # script existing on PATH. We also check for the module explicitly so
        # failures become actionable instead of cryptic infra errors.
        try:
            import pytest  # noqa: F401
        except ModuleNotFoundError:
            return SandboxResult(
                rc=-2,
                stdout="",
                stderr="pytest is not installed in the current environment. Run: pip install pytest",
                elapsed_seconds=0.0,
                mode="local",
                infra_error="pytest_not_installed",
            )

        pytest_cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short"] + self.extra_pytest_args
        start = time.monotonic()
        try:
            result = subprocess.run(
                pytest_cmd,
                cwd=path,
                capture_output=True,
                text=True,
                timeout=self.local_timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            elapsed = time.monotonic() - start
            logger.info(
                "Local tests finished: rc=%d elapsed=%.1fs", result.returncode, elapsed
            )
            return SandboxResult(
                rc=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                elapsed_seconds=elapsed,
                mode="local",
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            logger.warning("Local test run timed out after %.1fs", elapsed)
            return SandboxResult(
                rc=-1,
                stdout="",
                stderr=f"Tests timed out after {self.local_timeout}s.",
                elapsed_seconds=elapsed,
                mode="local",
                timed_out=True,
                infra_error="timeout",
            )
        except FileNotFoundError:
            return SandboxResult(
                rc=-2,
                stdout="",
                stderr="pytest not found. Is it installed in this environment?",
                mode="local",
                infra_error="pytest_not_found",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected local sandbox error")
            return SandboxResult(
                rc=-2,
                stdout="",
                stderr=str(exc),
                mode="local",
                infra_error=str(exc),
            )

    # ── Docker ───────────────────────────────────────────────

    def _run_docker(self, path: Path) -> SandboxResult:
        if not self._docker_available():
            return SandboxResult(
                rc=-2,
                stdout="",
                stderr="Docker is not available on this system.",
                mode="docker",
                infra_error="docker_not_found",
            )

        pytest_args = " ".join(["-q", "--tb=short"] + self.extra_pytest_args)
        bash_cmd = (
            "pip install --quiet pytest 2>&1 || true && "
            "pip install --quiet -r requirements.txt 2>&1 || true && "
            f"pytest {pytest_args}"
        )

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--cpus", "1.0",
            "--memory", "512m",
            "--read-only",
            "--tmpfs", "/tmp",
            "--tmpfs", "/root",
            "-v", f"{path}:/app:ro",
            "-w", "/app",
            self.docker_image,
            "bash", "-c", bash_cmd,
        ]

        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.docker_timeout,
            )
            elapsed = time.monotonic() - start
            logger.info(
                "Docker tests finished: rc=%d elapsed=%.1fs", result.returncode, elapsed
            )
            return SandboxResult(
                rc=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                elapsed_seconds=elapsed,
                mode="docker",
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start
            logger.warning("Docker test run timed out after %.1fs", elapsed)
            return SandboxResult(
                rc=-1,
                stdout="",
                stderr=f"Docker run timed out after {self.docker_timeout}s.",
                elapsed_seconds=elapsed,
                mode="docker",
                timed_out=True,
                infra_error="timeout",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected Docker sandbox error")
            return SandboxResult(
                rc=-2,
                stdout="",
                stderr=str(exc),
                mode="docker",
                infra_error=str(exc),
            )

    @staticmethod
    def _docker_available() -> bool:
        try:
            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
                check=True,
            )
            return True
        except Exception:
            return False