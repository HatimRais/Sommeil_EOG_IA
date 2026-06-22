"""
DeepSleep AI — analysis service (shared logic for Streamlit + FastAPI).
"""
from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
import openvino as ov
import pandas as pd

from src.data_loader import load_and_sync_labels
from src.preprocessing import apply_preprocessing

EPOCH_DURATION = 30.0
SFREQ = 100
N_POINTS = int(EPOCH_DURATION * SFREQ)
BATCH_INFER = 64
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_FULL = [
    "Wake",
    "N1 (Stage 1)",
    "N2 (Stage 2)",
    "N3 (Slow Wave Sleep)",
    "REM",
]

NORMS: Dict[str, Tuple[float, float, str]] = {
    "TST_min": (390, 480, "Total Sleep Time"),
    "SE_pct": (85, 100, "Sleep Efficiency"),
    "Latency_min": (0, 20, "Sleep Onset Latency"),
    "WASO_min": (0, 30, "Wake After Sleep Onset"),
    "REM_pct": (20, 25, "REM %"),
    "N1_pct": (2, 5, "N1 %"),
    "N2_pct": (45, 55, "N2 %"),
    "N3_pct": (13, 23, "N3 (SWS) %"),
    "REM_lat_min": (70, 120, "REM Latency"),
    "Awakenings": (0, 10, "# Awakenings"),
}

DEVICE_LABELS = {
    "NPU": "Intel AI Boost (NPU)",
    "GPU": "Integrated GPU",
    "CPU": "CPU",
    "AUTO": "Auto-select",
}

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_XML = os.path.join(_PROJECT_ROOT, "models", "sleep_model_npu.xml")
DATA_RAW = os.path.join(_PROJECT_ROOT, "data", "raw")


def list_devices() -> List[str]:
    try:
        return list(ov.Core().available_devices)
    except Exception:
        return []


@lru_cache(maxsize=4)
def load_engine(device: str) -> Tuple[Any, str, Optional[str]]:
    if not os.path.exists(MODEL_XML):
        return None, "Model file not found", None
    try:
        core = ov.Core()
        m = core.read_model(MODEL_XML)
        m.reshape({m.inputs[0]: [BATCH_INFER, N_POINTS, 1]})
        net = core.compile_model(m, device)
        try:
            exec_dev = "".join(list(net.get_property("EXECUTION_DEVICES")))
        except Exception:
            exec_dev = device
        try:
            base = exec_dev.split(".")[0]
            hw = core.get_property(base, "FULL_DEVICE_NAME")
        except Exception:
            hw = exec_dev
        return net, exec_dev, hw
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}", None


def infer_batched(net, X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    pad = (-n) % BATCH_INFER
    if pad:
        X = np.concatenate([X, np.zeros((pad,) + X.shape[1:], dtype=X.dtype)], axis=0)
    outs, key = [], net.output(0)
    for i in range(0, X.shape[0], BATCH_INFER):
        outs.append(net(X[i : i + BATCH_INFER])[key])
    return np.concatenate(outs, axis=0)[:n]


def anonymous_id(path: str) -> str:
    return "PSG-" + hashlib.md5(path.encode()).hexdigest()[:8].upper()


def fmt_hms(minutes: float) -> str:
    if not np.isfinite(minutes):
        return "—"
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h {m:02d}min"


def clinical_status(value, lo, hi, lower_better: bool = False) -> Tuple[str, str]:
    if value is None or not np.isfinite(value):
        return "n/a", "info"
    if lower_better:
        if value <= hi:
            return "NORMAL", "norm"
        if value <= hi * 1.5:
            return "BORDERLINE", "warn"
        return "ELEVATED", "alert"
    if lo <= value <= hi:
        return "NORMAL", "norm"
    if value < lo:
        if value >= lo * 0.85:
            return "LOW", "warn"
        return "LOW", "alert"
    if value <= hi * 1.15:
        return "HIGH", "warn"
    return "HIGH", "alert"


def detect_sleep_cycles(preds: np.ndarray) -> int:
    cycles = 0
    i = 0
    rem_min_epochs = 2
    while i < len(preds):
        if preds[i] == 4:
            run = 0
            while i < len(preds) and preds[i] == 4:
                run += 1
                i += 1
            if run >= rem_min_epochs:
                cycles += 1
        else:
            i += 1
    return cycles


def compute_clinical_report(preds: np.ndarray, epoch_dur: float = EPOCH_DURATION) -> dict:
    n = len(preds)
    tib = n * epoch_dur / 60.0
    sleep = preds != 0
    tst = sleep.sum() * epoch_dur / 60.0
    se = (tst / tib * 100) if tib else 0

    sleep_idx = np.where(sleep)[0]
    if len(sleep_idx) > 0:
        first_sleep = sleep_idx[0]
        sol = first_sleep * epoch_dur / 60.0
        post_sol = preds[first_sleep:]
        waso = (post_sol == 0).sum() * epoch_dur / 60.0
        rem_idx = np.where(post_sol == 4)[0]
        rem_lat = rem_idx[0] * epoch_dur / 60.0 if len(rem_idx) else float("nan")
    else:
        sol = float("nan")
        waso = 0
        rem_lat = float("nan")

    counts = {s: int((preds == i).sum()) for i, s in enumerate(STAGES)}
    sleep_total = max(1, sleep.sum())
    pct = {
        "W_pct": counts["W"] / max(1, n) * 100,
        "N1_pct": counts["N1"] / sleep_total * 100,
        "N2_pct": counts["N2"] / sleep_total * 100,
        "N3_pct": counts["N3"] / sleep_total * 100,
        "REM_pct": counts["REM"] / sleep_total * 100,
    }
    awakenings = int(np.sum((preds[:-1] != 0) & (preds[1:] == 0)))
    transitions = int((np.diff(preds) != 0).sum())
    fragmentation = transitions / max(1, sleep.sum()) * 100

    return {
        "TIB_min": float(tib),
        "TST_min": float(tst),
        "SE_pct": float(se),
        "Latency_min": float(sol) if np.isfinite(sol) else None,
        "WASO_min": float(waso),
        "REM_lat_min": float(rem_lat) if np.isfinite(rem_lat) else None,
        "Awakenings": awakenings,
        "Transitions": transitions,
        "Fragmentation_idx": float(fragmentation),
        "Cycles": detect_sleep_cycles(preds),
        "Counts": counts,
        **{k: float(v) for k, v in pct.items()},
    }


def cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = y_true != -1
    yt, yp = y_true[valid], y_pred[valid]
    n = len(yt)
    if n == 0:
        return float("nan")
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    po = np.trace(cm) / n
    pe = sum(cm[i].sum() * cm[:, i].sum() for i in range(5)) / (n**2)
    if pe == 1:
        return 1.0
    return float((po - pe) / (1 - pe))


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> List[dict]:
    valid = y_true != -1
    yt, yp = y_true[valid], y_pred[valid]
    rows = []
    for i, name in enumerate(STAGES):
        tp = int(((yt == i) & (yp == i)).sum())
        fp = int(((yt != i) & (yp == i)).sum())
        fn = int(((yt == i) & (yp != i)).sum())
        sup = int((yt == i).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        rows.append(
            {
                "stage": name,
                "sensitivity": round(rec * 100, 1),
                "precision": round(prec * 100, 1),
                "f1": round(f1 * 100, 1),
                "support": sup,
            }
        )
    return rows


def confusion_matrix_data(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    valid = y_true != -1
    yt, yp = y_true[valid], y_pred[valid]
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    return {
        "matrix": cm.tolist(),
        "matrixNorm": cm_norm.round(1).tolist(),
        "stages": STAGES,
    }


def build_interpretation(report: dict) -> List[dict]:
    interp = []
    se = report["SE_pct"]
    if se >= 85:
        interp.append({"text": f"Sleep efficiency is normal (≥85%).", "severity": "norm"})
    elif se >= 75:
        interp.append({"text": f"Sleep efficiency is borderline ({se:.1f}%).", "severity": "warn"})
    else:
        interp.append(
            {
                "text": f"Sleep efficiency is reduced ({se:.1f}%) — possible insomnia or fragmented sleep.",
                "severity": "alert",
            }
        )

    lat = report.get("Latency_min")
    if lat is not None and np.isfinite(lat):
        if lat < 5:
            interp.append(
                {
                    "text": f"Very short sleep onset latency ({lat:.1f} min) — suggests excessive sleepiness.",
                    "severity": "warn",
                }
            )
        elif lat > 30:
            interp.append(
                {
                    "text": f"Prolonged sleep onset latency ({lat:.1f} min) — possible insomnia.",
                    "severity": "warn",
                }
            )

    rem = report["REM_pct"]
    if rem < 15:
        interp.append(
            {"text": f"REM sleep is reduced ({rem:.1f}% of TST). Consider REM suppression.", "severity": "warn"}
        )
    elif rem > 30:
        interp.append({"text": f"REM sleep is elevated ({rem:.1f}% of TST).", "severity": "warn"})
    else:
        interp.append({"text": f"REM sleep proportion is within normal range ({rem:.1f}%).", "severity": "norm"})

    n3 = report["N3_pct"]
    if n3 < 10:
        interp.append(
            {"text": f"Slow-wave sleep (N3) is markedly reduced ({n3:.1f}%).", "severity": "alert"}
        )
    elif n3 < 13:
        interp.append({"text": f"Slow-wave sleep (N3) is low ({n3:.1f}%).", "severity": "warn"})
    else:
        interp.append({"text": f"Slow-wave sleep (N3) is within expected range ({n3:.1f}%).", "severity": "norm"})

    if report["Awakenings"] > 15:
        interp.append(
            {
                "text": f"High number of awakenings ({report['Awakenings']}) — indicates fragmented sleep.",
                "severity": "alert",
            }
        )

    if report["Cycles"] < 3:
        interp.append(
            {
                "text": f"Only {report['Cycles']} sleep cycle(s) detected — typically 4–6 in adults.",
                "severity": "warn",
            }
        )

    return interp


def make_report_csv(preds: np.ndarray, y_true: Optional[np.ndarray] = None) -> bytes:
    df = pd.DataFrame(
        {
            "epoch": np.arange(len(preds)),
            "time_hms": [
                f"{int(i * EPOCH_DURATION // 3600):02d}:"
                f"{int((i * EPOCH_DURATION % 3600) // 60):02d}:"
                f"{int(i * EPOCH_DURATION % 60):02d}"
                for i in range(len(preds))
            ],
            "stage_AI": [STAGES[p] for p in preds],
        }
    )
    if y_true is not None:
        df["stage_expert"] = [STAGES[t] if t >= 0 else "?" for t in y_true]
        df["agreement"] = [
            "OK" if (t >= 0 and p == t) else ("MISMATCH" if t >= 0 else "")
            for p, t in zip(preds, y_true)
        ]
    # Séparateur ; + BOM UTF-8 pour Excel (locale française Windows)
    return df.to_csv(index=False, sep=";").encode("utf-8-sig")


def list_patients() -> List[dict]:
    if not os.path.isdir(DATA_RAW):
        return []
    patients = []
    for f in sorted(os.listdir(DATA_RAW)):
        if "Signal.edf" in f:
            lab = f.replace("Signal.edf", "Labels.edf")
            patients.append(
                {
                    "id": f.replace("_Signal.edf", ""),
                    "signalFile": f,
                    "hasLabels": os.path.exists(os.path.join(DATA_RAW, lab)),
                }
            )
    return patients


def _json_safe(val):
    if isinstance(val, (np.floating, float)):
        return None if not np.isfinite(val) else float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, dict):
        return {k: _json_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_json_safe(v) for v in val]
    return val


def analyze_recording(
    sig_path: str,
    lab_path: Optional[str] = None,
    device: str = "AUTO",
    source_name: Optional[str] = None,
) -> dict:
    net, exec_dev, hw_name = load_engine(device)
    if net is None:
        raise RuntimeError(exec_dev)

    raw = mne.io.read_raw_edf(sig_path, preload=True, verbose=False)
    sfreq_orig = raw.info["sfreq"]
    duration_min = raw.times[-1] / 60.0
    n_channels = len(raw.ch_names)
    meas_date = raw.info.get("meas_date")

    data, eog_ch = apply_preprocessing(raw)
    n_epochs = len(data) // N_POINTS
    if n_epochs == 0:
        raise ValueError("Recording too short (no full 30-s epoch).")

    X = data[: n_epochs * N_POINTS].reshape(n_epochs, N_POINTS, 1).astype(np.float32)

    t0 = time.time()
    raw_out = infer_batched(net, X)
    preds = np.argmax(raw_out, axis=1)
    confs = np.max(raw_out, axis=1)
    inf_t = time.time() - t0

    y_true = None
    if lab_path and os.path.exists(lab_path):
        y_true = load_and_sync_labels(lab_path, n_epochs)

    report = compute_clinical_report(preds)
    patient_id = anonymous_id(sig_path)
    rec_date = meas_date.strftime("%Y-%m-%d %H:%M") if meas_date else "unknown"
    basename = source_name or os.path.basename(sig_path)

    validation = None
    if y_true is not None:
        valid = y_true != -1
        has_valid = not np.all(y_true == -1)
        if has_valid:
            ko = cohens_kappa(y_true, preds)
            validation = {
                "accuracy": round(float((preds[valid] == y_true[valid]).mean() * 100), 2),
                "kappa": round(float(ko), 3) if np.isfinite(ko) else None,
                "kappaLabel": (
                    "Almost perfect"
                    if ko >= 0.81
                    else "Substantial"
                    if ko >= 0.61
                    else "Moderate"
                    if ko >= 0.41
                    else "Fair"
                    if ko >= 0.21
                    else "Poor"
                ),
                "epochsEvaluated": int(valid.sum()),
                "perStage": per_class_metrics(y_true, preds),
                "confusion": confusion_matrix_data(y_true, preds),
                "expertStages": [
                    STAGES[int(t)] if t >= 0 else None for t in y_true.tolist()
                ],
            }
        else:
            validation = {"parseError": True}

    metrics_with_status = []
    metric_defs = [
        ("Total Sleep Time", "TST_min", "TST_min", False, "time"),
        ("Sleep Efficiency", "SE_pct", "SE_pct", False, "pct"),
        ("Sleep Latency", "Latency_min", "Latency_min", True, "time"),
        ("WASO", "WASO_min", "WASO_min", True, "time"),
        ("REM Latency", "REM_lat_min", "REM_lat_min", False, "time"),
        ("REM %", "REM_pct", "REM_pct", False, "pct"),
        ("N3 (SWS) %", "N3_pct", "N3_pct", False, "pct"),
        ("N2 %", "N2_pct", "N2_pct", False, "pct"),
        ("Awakenings", "Awakenings", "Awakenings", True, "count"),
    ]
    for label, key, norm_key, lower_better, fmt in metric_defs:
        val = report[key]
        lo, hi, _ = NORMS[norm_key]
        status_label, status = clinical_status(val, lo, hi, lower_better=lower_better)
        if fmt == "time":
            display = fmt_hms(val) if val is not None else "—"
        elif fmt == "pct":
            display = f"{val:.1f} %" if val is not None else "—"
        else:
            display = str(int(val)) if val is not None else "—"
        metrics_with_status.append(
            {
                "label": label,
                "value": display,
                "raw": _json_safe(val),
                "status": status,
                "statusLabel": status_label,
                "reference": f"{lo}–{hi}",
            }
        )

    predictions = [
        {
            "epoch": i,
            "stage": STAGES[int(p)],
            "stageIndex": int(p),
            "confidence": round(float(c) * 100, 1),
            "timeHours": round(i * EPOCH_DURATION / 3600.0, 4),
        }
        for i, (p, c) in enumerate(zip(preds, confs))
    ]

    result = {
        "patient": {
            "id": patient_id,
            "recordingDate": rec_date,
            "durationMin": round(float(duration_min), 1),
            "durationFormatted": fmt_hms(duration_min),
            "sourceFile": basename,
            "eogChannel": eog_ch,
            "sfreqOrig": round(float(sfreq_orig), 0),
            "sfreqTarget": SFREQ,
            "nEpochs": n_epochs,
            "inferenceMs": round(inf_t * 1000, 0),
        },
        "predictions": predictions,
        "clinical": report,
        "metrics": metrics_with_status,
        "interpretation": build_interpretation(report),
        "validation": validation,
        "engine": {
            "device": device,
            "runtimeDevice": exec_dev,
            "hardware": hw_name,
            "inferenceMs": round(inf_t * 1000, 1),
            "throughput": round(n_epochs / inf_t, 0) if inf_t > 0 else 0,
            "meanConfidence": round(float(confs.mean() * 100), 1),
        },
        "technical": {
            "channels": n_channels,
            "bandpass": "FIR 0.5 – 35 Hz",
            "normalization": "Z-score + clip ±3σ",
            "epochLength": f"{int(EPOCH_DURATION)} s ({N_POINTS} samples)",
            "model": "1D-CNN (NPU-compatible)",
            "format": "OpenVINO IR FP16",
            "parameters": "455 557",
            "inputShape": f"({BATCH_INFER}, {N_POINTS}, 1)",
        },
        "stages": {
            "short": STAGES,
            "full": STAGE_FULL,
        },
        "cycles": report["Cycles"],
        "meanConfidence": round(float(confs.mean() * 100), 1),
        "fragmentation": report["Fragmentation_idx"],
        "transitions": report["Transitions"],
    }
    return _json_safe(result)
