"""Inicializa o ambiente local completo da Pokédex Inteligente."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def run(command: list[str], cwd: Path, label: str) -> None:
    print(f"\n[{label}] {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, env=os.environ.copy())
    if result.returncode != 0:
        raise SystemExit(f"Falha em {label} (código {result.returncode}).")


def artifacts_ready() -> bool:
    result = subprocess.run([str(PYTHON), "-m", "app.cli", "verify"], cwd=BACKEND, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
        return True
    print(result.stdout.strip() or result.stderr.strip())
    return False


def start_process(command: list[str], cwd: Path, label: str):
    print(f"[{label}] iniciando: {' '.join(command)}", flush=True)
    return subprocess.Popen(command, cwd=cwd, env=os.environ.copy(), creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)


def main() -> None:
    print("Pokédex Inteligente — inicialização local", flush=True)
    if not artifacts_ready():
        print("Artefatos ausentes ou inválidos. Executando notebooks em sequência...", flush=True)
        run([str(PYTHON), "scripts/run_notebooks.py"], ROOT, "pipeline")
        if not artifacts_ready():
            raise SystemExit("O pipeline terminou, mas os artefatos continuam inválidos.")

    if not shutil.which("npm"):
        raise SystemExit("Node.js/npm não encontrado. Instale Node.js antes de iniciar o frontend.")
    if not (FRONTEND / "node_modules").exists():
        run(["npm.cmd", "install"], FRONTEND, "frontend dependencies")

    backend = start_process([str(PYTHON), "-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8000"], BACKEND, "backend")
    frontend = start_process(["npm.cmd", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"], FRONTEND, "frontend")
    print("\nBackend:  http://127.0.0.1:8000/docs", flush=True)
    print("Frontend: http://127.0.0.1:5173", flush=True)
    print("Pressione Ctrl+C para encerrar os processos.", flush=True)
    try:
        while True:
            if backend.poll() is not None:
                raise SystemExit("O backend encerrou inesperadamente.")
            if frontend.poll() is not None:
                raise SystemExit("O frontend encerrou inesperadamente.")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nEncerrando backend e frontend...", flush=True)
        for process in (frontend, backend):
            if process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    main()

