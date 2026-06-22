"""Vérifie la présence des modèles OpenVINO avant le démarrage Railway."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = ROOT / "models" / "sleep_model_npu.xml"
MODEL_BIN = ROOT / "models" / "sleep_model_npu.bin"


def verify() -> None:
    missing = [p for p in (MODEL_XML, MODEL_BIN) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Modèles OpenVINO introuvables dans l'image Docker:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )
    print(f"[deepsleep] Modèles OK — {MODEL_XML.name}, {MODEL_BIN.name}")


if __name__ == "__main__":
    verify()
