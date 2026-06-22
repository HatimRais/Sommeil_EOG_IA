"""Point d'entrée Railway : API FastAPI interne + site Next.js (modèle SICAM)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_inference_models import verify  # noqa: E402

API_PORT = os.environ.get("DEEPSLEEP_INTERNAL_API_PORT", "8001")
WEB_PORT = os.environ.get("PORT", "3000")
API_URL = f"http://127.0.0.1:{API_PORT}"

_procs: list[subprocess.Popen] = []


def _shutdown(*_args) -> None:
    for proc in _procs:
        if proc.poll() is None:
            proc.terminate()
    for proc in _procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _wait_api(timeout_sec: int = 180) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{API_URL}/api/health", timeout=5) as resp:
                if resp.status == 200:
                    print(f"[deepsleep] API prête sur {API_URL}")
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise RuntimeError(f"API non joignable sur {API_URL} après {timeout_sec}s")


def main() -> None:
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    verify()
    os.environ["DEEPSLEEP_INFERENCE_URL"] = API_URL

    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        API_PORT,
        "--workers",
        "1",
    ]
    print(f"[deepsleep] Démarrage API interne : {API_URL}")
    api_proc = subprocess.Popen(api_cmd)
    _procs.append(api_proc)

    _wait_api()

    web_env = {
        **os.environ,
        "PORT": WEB_PORT,
        "HOSTNAME": "0.0.0.0",
        "DEEPSLEEP_INFERENCE_URL": API_URL,
    }
    print(f"[deepsleep] Démarrage Next.js sur 0.0.0.0:{WEB_PORT}")
    web_proc = subprocess.Popen(
        ["node", "server.js"], cwd=ROOT / "frontend", env=web_env
    )
    _procs.append(web_proc)

    sys.exit(web_proc.wait())


if __name__ == "__main__":
    main()
