import os
import subprocess
import shutil

class Sandbox:
    def __init__(self, mode="local"):
        self.mode = mode

    def run_tests(self, project_path):
        if self.mode == "docker":
            return self._run_docker(project_path)
        else:
            return self._run_local(project_path)
            
    def _run_local(self, project_path):
        print(f"Running pytest locally in {project_path}")
        try:
            result = subprocess.run(
                ["pytest", "-q"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            return {"rc": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired as e:
            return {"rc": -1, "stdout": "Timeout", "stderr": "Tests timed out"}
        except Exception as e:
            return {"rc": -2, "stdout": "", "stderr": str(e)}

    def _run_docker(self, project_path):
        print(f"Running pytest in Docker for {project_path}")
        import tempfile
        abs_path = os.path.abspath(project_path)
        
        cmd = [
            "docker", "run", "--rm", "--network", "none",
            "--cpus", "0.5", "--memory", "512m",
            "-v", f"{abs_path}:/app", "-w", "/app",
            "python:3.10-slim",
            "bash", "-c", "pip install -r requirements.txt && pytest -q"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return {"rc": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"rc": -2, "stdout": "", "stderr": str(e)}