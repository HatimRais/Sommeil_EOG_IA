"""
DeepSleep AI — FastAPI REST backend for Next.js frontend.
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.api.service import (  # noqa: E402
    DATA_RAW,
    DEVICE_LABELS,
    NORMS,
    analyze_recording,
    list_devices,
    list_patients,
    make_report_csv,
)

app = FastAPI(
    title="DeepSleep AI API",
    description="EOG-based AASM 5-class sleep staging",
    version="2.0.0",
)

def _cors_origins() -> list[str]:
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ]
    extra = os.getenv("CORS_ORIGINS", "")
    if extra:
        origins.extend(o.strip() for o in extra.split(",") if o.strip())
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"https://[\w-]+(\.up\.railway\.app|\.vercel\.app)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "DeepSleep AI"}


@app.get("/api/devices")
def devices():
    available = list_devices()
    options = [d for d in ["NPU", "GPU", "CPU"] if d in available]
    if not options:
        options = available[:]
    options.append("AUTO")
    return {
        "available": available,
        "options": [
            {"id": d, "label": DEVICE_LABELS.get(d, d)} for d in options
        ],
    }


@app.get("/api/patients")
def patients():
    return {"patients": list_patients()}


@app.get("/api/norms")
def norms():
    return {
        "norms": [
            {
                "key": k,
                "low": lo,
                "high": hi,
                "label": label,
                "unit": "min"
                if "min" in k or "Latency" in k
                else ("%" if "pct" in k else ""),
            }
            for k, (lo, hi, label) in NORMS.items()
        ]
    }


@app.post("/api/analyze")
async def analyze(
    device: str = Form("AUTO"),
    signal: Optional[UploadFile] = File(None),
    labels: Optional[UploadFile] = File(None),
    patient_id: Optional[str] = Form(None),
):
    sig_path = None
    lab_path = None
    source_name = None
    tmp_files: list[str] = []

    try:
        if patient_id:
            sig_file = f"{patient_id}_Signal.edf"
            sig_path = os.path.join(DATA_RAW, sig_file)
            if not os.path.exists(sig_path):
                raise HTTPException(404, f"Patient record not found: {patient_id}")
            source_name = sig_file
            lab_candidate = os.path.join(DATA_RAW, f"{patient_id}_Labels.edf")
            if os.path.exists(lab_candidate):
                lab_path = lab_candidate
        elif signal:
            suffix = ".edf"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            content = await signal.read()
            tmp.write(content)
            tmp.close()
            sig_path = tmp.name
            tmp_files.append(sig_path)
            source_name = signal.filename
        else:
            raise HTTPException(400, "Provide signal file or patient_id")

        if labels:
            tmp_lab = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
            tmp_lab.write(await labels.read())
            tmp_lab.close()
            lab_path = tmp_lab.name
            tmp_files.append(lab_path)

        result = analyze_recording(
            sig_path,
            lab_path=lab_path,
            device=device,
            source_name=source_name,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except OSError:
                pass


@app.post("/api/export/csv")
async def export_csv(
    signal: Optional[UploadFile] = File(None),
    labels: Optional[UploadFile] = File(None),
    patient_id: Optional[str] = Form(None),
    device: str = Form("AUTO"),
):
    """Re-run analysis and return epoch CSV."""
    import numpy as np

    sig_path = None
    lab_path = None
    tmp_files: list[str] = []

    try:
        if patient_id:
            sig_path = os.path.join(DATA_RAW, f"{patient_id}_Signal.edf")
            lab_candidate = os.path.join(DATA_RAW, f"{patient_id}_Labels.edf")
            if os.path.exists(lab_candidate):
                lab_path = lab_candidate
        elif signal:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
            tmp.write(await signal.read())
            tmp.close()
            sig_path = tmp.name
            tmp_files.append(sig_path)
        else:
            raise HTTPException(400, "Provide signal or patient_id")

        if labels:
            tmp_lab = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
            tmp_lab.write(await labels.read())
            tmp_lab.close()
            lab_path = tmp_lab.name
            tmp_files.append(lab_path)

        result = analyze_recording(sig_path, lab_path=lab_path, device=device)
        preds = np.array([p["stageIndex"] for p in result["predictions"]])
        y_true = None
        if result.get("validation") and result["validation"].get("expertStages"):
            y_true = np.array(
                [
                    -1 if s is None else result["stages"]["short"].index(s)
                    for s in result["validation"]["expertStages"]
                ]
            )
        csv_content = make_report_csv(preds, y_true)
        filename = f"{result['patient']['id']}_hypnogram.csv"
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except OSError:
                pass


def _mount_frontend(app: FastAPI) -> None:
    """Sert le frontend Next.js exporté (frontend/out) sur / — une seule URL Railway."""
    static_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "out")
    )
    if not os.path.isdir(static_dir):
        return

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    index_html = os.path.join(static_dir, "index.html")
    next_dir = os.path.join(static_dir, "_next")

    if os.path.isdir(next_dir):
        app.mount("/_next", StaticFiles(directory=next_dir), name="next-assets")

    @app.get("/", include_in_schema=False)
    async def spa_root():
        if os.path.isfile(index_html):
            return FileResponse(index_html, media_type="text/html")
        raise HTTPException(503, "Frontend non buildé (frontend/out manquant)")

    @app.get("/{asset_path:path}", include_in_schema=False)
    async def spa_fallback(asset_path: str):
        if asset_path.startswith("api/") or asset_path in ("docs", "openapi.json", "redoc"):
            raise HTTPException(404, detail="Not Found")
        file_path = os.path.join(static_dir, asset_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        if os.path.isfile(index_html):
            return FileResponse(index_html, media_type="text/html")
        raise HTTPException(404, detail="Not Found")


_mount_frontend(app)
