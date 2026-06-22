"""
Génère toutes les figures du rapport (PNG) à partir des données réelles du projet.

Sorties : reports/figures/*.png
Utilise : Patient_01 (data/raw), le modèle OpenVINO et les métriques JSON.

Usage :
  .\.venv\Scripts\python.exe scripts/make_report_assets.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

FIG_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_COLORS = ["#EAB308", "#93C5E8", "#3B82B6", "#0C4A6E", "#B91C1C"]
PRIMARY = "#0C4A6E"
ACCENT = "#1E5F8A"
N_POINTS = 3000
BATCH = 64

# Courbe d'apprentissage (val F1 macro) — journal Kaggle CNN 1D NPU (epochs 8 → 25)
TRAIN_EPOCHS = list(range(8, 26))
TRAIN_VAL_F1 = [
    0.6086, 0.6329, 0.6927, 0.7067, 0.7081, 0.6837, 0.7094, 0.7120,
    0.7127, 0.7085, 0.7244, 0.7085, 0.7110, 0.7103, 0.6966, 0.7096,
    0.6967, 0.7131,
]
TRAIN_VAL_ACC = [
    0.7629, 0.7702, 0.8165, 0.8340, 0.8336, 0.8063, 0.8396, 0.8301,
    0.8349, 0.8328, 0.8424, 0.8296, 0.8305, 0.8362, 0.8230, 0.8336,
    0.8214, 0.8334,
]


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  +", os.path.relpath(path, ROOT))
    return path


def load_json(name):
    with open(os.path.join(ROOT, "models", name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Inférence Patient_01 (réutilisée par plusieurs figures)
# ---------------------------------------------------------------------------
def run_patient01():
    import mne
    import openvino as ov
    from src.preprocessing import apply_preprocessing
    from src.data_loader import load_and_sync_labels

    sig = os.path.join(ROOT, "data", "raw", "Patient_01_Signal.edf")
    lab = os.path.join(ROOT, "data", "raw", "Patient_01_Labels.edf")

    raw = mne.io.read_raw_edf(sig, preload=True, verbose=False)
    if raw.info["sfreq"] != 100.0:
        raw_raw = raw.copy().resample(100.0)
    else:
        raw_raw = raw.copy()
    eog_ch = [c for c in raw.ch_names if "EOG" in c.upper()][0]
    raw_signal = raw_raw.get_data(picks=[eog_ch])[0]  # avant filtrage/normalisation

    data, eog = apply_preprocessing(raw)  # filtré + z-score + clip
    n_epochs = len(data) // N_POINTS
    X = data[: n_epochs * N_POINTS].reshape(n_epochs, N_POINTS, 1).astype(np.float32)

    core = ov.Core()
    m = core.read_model(os.path.join(ROOT, "models", "sleep_model_npu.xml"))
    m.reshape({m.inputs[0]: [BATCH, N_POINTS, 1]})
    net = core.compile_model(m, "CPU")
    pad = (-n_epochs) % BATCH
    Xp = np.concatenate([X, np.zeros((pad, N_POINTS, 1), np.float32)]) if pad else X
    key = net.output(0)
    out = np.concatenate([net(Xp[i : i + BATCH])[key] for i in range(0, Xp.shape[0], BATCH)])[:n_epochs]
    preds = np.argmax(out, axis=1)
    confs = np.max(out, axis=1)

    y_true = load_and_sync_labels(lab, n_epochs)
    return {
        "raw_signal": raw_signal[: n_epochs * N_POINTS],
        "proc_signal": data[: n_epochs * N_POINTS],
        "X": X,
        "preds": preds,
        "confs": confs,
        "y_true": y_true,
        "eog_ch": eog,
        "n_epochs": n_epochs,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_eog_signal(p):
    """Signal EOG brut vs prétraité sur une époque de 30 s (stade N2 si possible)."""
    y_true = p["y_true"]
    target = 2 if (y_true == 2).any() else int(np.bincount(y_true[y_true >= 0]).argmax())
    idx = int(np.where(y_true == target)[0][min(3, (y_true == target).sum() - 1)])
    s = idx * N_POINTS
    raw_ep = p["raw_signal"][s : s + N_POINTS]
    proc_ep = p["proc_signal"][s : s + N_POINTS]
    t = np.arange(N_POINTS) / 100.0

    fig, ax = plt.subplots(2, 1, figsize=(9.5, 4.4), sharex=True)
    ax[0].plot(t, raw_ep * 1e6 if np.max(np.abs(raw_ep)) < 1 else raw_ep, color=ACCENT, lw=0.8)
    ax[0].set_title(f"Signal EOG brut — époque {idx} (stade {STAGES[target]})", fontsize=10, color=PRIMARY)
    ax[0].set_ylabel("Amplitude (µV)")
    ax[0].grid(alpha=0.3)
    ax[1].plot(t, proc_ep, color=PRIMARY, lw=0.8)
    ax[1].set_title("Signal EOG prétraité (filtre 0,5–35 Hz · z-score · clip ±3σ)", fontsize=10, color=PRIMARY)
    ax[1].set_ylabel("Amplitude (z)")
    ax[1].set_xlabel("Temps (s)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_eog_signal.png")


def fig_eog_stages(p):
    """Une époque représentative pour chaque stade détecté."""
    y_true = p["y_true"]
    fig, axes = plt.subplots(5, 1, figsize=(9.5, 7.2), sharex=True)
    t = np.arange(N_POINTS) / 100.0
    for k in range(5):
        ax = axes[k]
        where = np.where(y_true == k)[0]
        if len(where):
            idx = int(where[len(where) // 2])
            seg = p["proc_signal"][idx * N_POINTS : idx * N_POINTS + N_POINTS]
            ax.plot(t, seg, color=STAGE_COLORS[k], lw=0.8)
        else:
            ax.text(0.5, 0.5, "absent", ha="center", va="center", transform=ax.transAxes, color="#999")
        ax.set_ylabel(STAGES[k], rotation=0, labelpad=18, fontsize=11, color=STAGE_COLORS[k], weight="bold")
        ax.set_ylim(-3.2, 3.2)
        ax.grid(alpha=0.25)
    axes[0].set_title("Morphologie du signal EOG prétraité par stade (Patient_01)", fontsize=11, color=PRIMARY)
    axes[-1].set_xlabel("Temps (s)")
    fig.tight_layout()
    return _save(fig, "fig_eog_stages.png")


def fig_hypnogram(p):
    """Hypnogramme IA vs expert pour Patient_01."""
    preds = p["preds"]
    y_true = p["y_true"]
    n = len(preds)
    hours = np.arange(n) * 30.0 / 3600.0
    order = [0, 4, 1, 2, 3]  # W, REM, N1, N2, N3 (axe vertical clinique)
    remap = {stage: pos for pos, stage in enumerate(order)}
    fig, ax = plt.subplots(2, 1, figsize=(9.8, 4.6), sharex=True)

    def draw(a, seq, title):
        valid = seq >= 0
        yv = np.array([remap.get(int(s), np.nan) for s in seq], dtype=float)
        a.step(hours[valid], yv[valid], where="post", color=PRIMARY, lw=0.9)
        a.set_yticks(range(5))
        a.set_yticklabels([STAGES[o] for o in order])
        a.set_title(title, fontsize=10, color=PRIMARY)
        a.grid(alpha=0.3)
        a.invert_yaxis()

    draw(ax[0], preds, "Hypnogramme prédit par l'IA (CNN 1D · NPU)")
    if y_true is not None and (y_true >= 0).any():
        draw(ax[1], y_true, "Hypnogramme de référence (scoring expert)")
    ax[1].set_xlabel("Temps (heures)")
    fig.tight_layout()
    return _save(fig, "fig_hypnogram.png")


def fig_confusion(metrics, name, title):
    cm = np.array(metrics["confusion_matrix"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(STAGES); ax.set_yticklabels(STAGES)
    ax.set_xlabel("Prédiction"); ax.set_ylabel("Vérité terrain")
    ax.set_title(title, fontsize=10, color=PRIMARY)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i, j]*100:.0f}%", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "#1f2937", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _save(fig, name)


def fig_class_distribution(metrics):
    supports = [metrics["per_class"][s]["support"] for s in STAGES]
    total = sum(supports)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars = ax.bar(STAGES, supports, color=STAGE_COLORS)
    for b, v in zip(bars, supports):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v/total*100:.1f}%",
                ha="center", va="bottom", fontsize=9, color="#334155")
    ax.set_ylabel("Nombre d'époques")
    ax.set_title("Distribution des stades — jeu de test hold-out (27 sujets)", fontsize=10, color=PRIMARY)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_class_distribution.png")


def fig_f1_comparison(cnn, bilstm):
    cnn_f1 = [cnn["per_class"][s]["f1"] for s in STAGES]
    bil_f1 = [bilstm["per_class"][s]["f1"] for s in STAGES]
    x = np.arange(5); w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.bar(x - w / 2, cnn_f1, w, label="CNN 1D (NPU)", color=PRIMARY)
    ax.bar(x + w / 2, bil_f1, w, label="CNN + Bi-LSTM", color="#94B8D0")
    ax.set_xticks(x); ax.set_xticklabels(STAGES)
    ax.set_ylabel("Score F1"); ax.set_ylim(0, 1)
    ax.set_title("F1 par stade — test hold-out", fontsize=10, color=PRIMARY)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_f1_comparison.png")


def fig_training_curve():
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.plot(TRAIN_EPOCHS, TRAIN_VAL_ACC, "-o", color=ACCENT, ms=3, label="Accuracy (val)")
    ax.plot(TRAIN_EPOCHS, TRAIN_VAL_F1, "-s", color=PRIMARY, ms=3, label="F1 macro (val)")
    best = int(np.argmax(TRAIN_VAL_F1))
    ax.scatter([TRAIN_EPOCHS[best]], [TRAIN_VAL_F1[best]], s=90, facecolors="none",
               edgecolors="#B91C1C", lw=1.6, zorder=5, label=f"Meilleur F1 = {TRAIN_VAL_F1[best]:.3f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Apprentissage CNN 1D NPU (validation par sujet)", fontsize=10, color=PRIMARY)
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "fig_training_curve.png")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11.0, 2.6))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 1.6); ax.axis("off")
    boxes = [
        (0.1, "Fichiers\nEDF"), (1.8, "Lecture\nMNE"), (3.5, "Prétraitement\nEOG"),
        (5.2, "Époques\n30 s · 3000 pts"), (6.9, "CNN 1D\n(Keras→OpenVINO)"),
        (8.6, "Inférence\nNPU/GPU/CPU"), (10.0, "Dashboard\nStreamlit"),
    ]
    w, h = 1.45, 0.95
    for i, (x, txt) in enumerate(boxes):
        ax.add_patch(mpatches.FancyBboxPatch((x, 0.35), w, h, boxstyle="round,pad=0.04",
                     facecolor="#E8F0F6", edgecolor=PRIMARY, linewidth=1.2))
        ax.text(x + w / 2, 0.82, txt, ha="center", va="center", fontsize=8, weight="600")
        if i < len(boxes) - 1:
            ax.annotate("", xy=(boxes[i + 1][0] - 0.04, 0.82), xytext=(x + w + 0.04, 0.82),
                        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.3))
    fig.tight_layout()
    return _save(fig, "fig_pipeline.png")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 2.3))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 1.2); ax.axis("off")
    layers = [
        (0.1, "Entrée\n3000×1"), (1.45, "Conv1D 64\nk=11·Pool4"),
        (2.95, "Conv1D 128\nk=7·Pool4"), (4.45, "Conv1D 256\nk=5·Pool4"),
        (5.95, "Conv1D 256\nk=3·Pool2"), (7.45, "GAP +\nDense 128"),
        (8.85, "Softmax\n5 stades"),
    ]
    w, h = 1.2, 0.6
    for i, (x, t) in enumerate(layers):
        ax.add_patch(mpatches.Rectangle((x, 0.35), w, h, facecolor="#DBEAFE",
                     edgecolor="#1E40AF", linewidth=1))
        ax.text(x + w / 2, 0.65, t, ha="center", va="center", fontsize=7, weight="600")
        if i < len(layers) - 1:
            ax.annotate("", xy=(layers[i + 1][0] - 0.02, 0.65), xytext=(x + w + 0.02, 0.65),
                        arrowprops=dict(arrowstyle="->", color="#64748B", lw=1))
    fig.tight_layout()
    return _save(fig, "fig_architecture.png")


def main():
    print("Génération des figures du rapport →", os.path.relpath(FIG_DIR, ROOT))
    cnn = load_json("metrics_holdout_test.json")
    bilstm = load_json("metrics_test_bilstm.json")

    fig_pipeline()
    fig_architecture()
    fig_training_curve()
    fig_class_distribution(cnn)
    fig_f1_comparison(cnn, bilstm)
    fig_confusion(cnn, "fig_confusion_cnn.png", "Matrice de confusion — CNN 1D NPU (test)")

    print("Inférence Patient_01 (OpenVINO CPU)…")
    p = run_patient01()
    acc = (p["preds"][p["y_true"] >= 0] == p["y_true"][p["y_true"] >= 0]).mean()
    print(f"  Patient_01 : {p['n_epochs']} époques · accuracy vs expert = {acc*100:.1f}%")
    fig_eog_signal(p)
    fig_eog_stages(p)
    fig_hypnogram(p)

    summary = {
        "patient01_epochs": int(p["n_epochs"]),
        "patient01_accuracy": float(acc),
        "patient01_channel": p["eog_ch"],
    }
    with open(os.path.join(FIG_DIR, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Terminé.")


if __name__ == "__main__":
    main()
