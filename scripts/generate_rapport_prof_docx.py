"""
Génère un rapport Word (synthèse projet) à la racine du dépôt.
Usage (depuis la racine du projet) :
  .\\.venv\\Scripts\\python.exe scripts/generate_rapport_prof_docx.py
Dépendance : pip install python-docx
"""
from __future__ import annotations

import os
import sys
import tempfile

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_NAME = "Rapport_Projet_Sommeil_EOG_IA.docx"


def _add_title(doc: Document, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    h.runs[0].font.color.rgb = RGBColor(12, 74, 110)


def _add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(10)
    for ri, row in enumerate(rows):
        cells = t.rows[ri + 1].cells
        for ci, val in enumerate(row):
            cells[ci].text = str(val)
            for p in cells[ci].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(10)


def _pipeline_figure(path: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 2.8))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 1.6)
    ax.axis("off")
    boxes = [
        (0.15, 0.35, "Fichiers\nEDF"),
        (1.85, 0.35, "Lecture\nMNE-Python"),
        (3.55, 0.35, "Prétraitement\nEOG"),
        (5.25, 0.35, "Découpage\népoques 30 s"),
        (6.95, 0.35, "Inférence\nOpenVINO IR"),
        (8.65, 0.35, "Dashboard\nStreamlit"),
    ]
    w, h = 1.55, 0.95
    for i, (x, y, txt) in enumerate(boxes):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04",
                facecolor="#E8F0F6",
                edgecolor="#0C4A6E",
                linewidth=1.2,
            )
        )
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5, weight="600")
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=(boxes[i + 1][0] - 0.05, y + h / 2),
                xytext=(x + w + 0.05, y + h / 2),
                arrowprops=dict(arrowstyle="->", color="#334155", lw=1.4),
            )
    fig.text(0.5, 0.08, "Chaîne de traitement — Sommeil_EOG_IA", ha="center", fontsize=9, color="#475569")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _architecture_figure(path: str) -> None:
    """Schéma simplifié CNN 1D (blocs)."""
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    layers = [
        (0.1, 0.35, "Entrée\n3000×1"),
        (1.35, 0.35, "Conv1D\n+ BN + Pool"),
        (2.85, 0.35, "Conv1D\n+ BN + Pool"),
        (4.35, 0.35, "Conv1D\n+ BN + Pool"),
        (5.85, 0.35, "Conv1D\n+ BN + Pool"),
        (7.35, 0.35, "GAP +\nDense"),
        (8.75, 0.35, "Softmax\n5 classes"),
    ]
    w = 1.1
    h = 0.55
    for i, (x, y, t) in enumerate(layers):
        ax.add_patch(
            mpatches.Rectangle((x, y), w, h, facecolor="#DBEAFE", edgecolor="#1E40AF", linewidth=1)
        )
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5, weight="600")
        if i < len(layers) - 1:
            ax.annotate(
                "",
                xy=(layers[i + 1][0] - 0.02, y + h / 2),
                xytext=(x + w + 0.02, y + h / 2),
                arrowprops=dict(arrowstyle="->", color="#64748B", lw=1),
            )
    fig.text(0.5, 0.06, "Architecture indicative — CNN 1D NPU (détail dans architecture.py)", ha="center", fontsize=8, color="#64748B")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_document() -> str:
    doc = Document()
    for s in doc.sections:
        s.top_margin = Pt(56)
        s.bottom_margin = Pt(56)
        s.left_margin = Pt(72)
        s.right_margin = Pt(72)

    # --- Page de titre ---
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Rapport de projet\n")
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = RGBColor(12, 74, 110)
    r = p.add_run("Classification automatique des stades du sommeil\n")
    r.bold = True
    r.font.size = Pt(16)
    r = p.add_run("à partir du signal EOG (électro-oculographie)\n")
    r.font.size = Pt(14)
    doc.add_paragraph()
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run("Projet : Sommeil_EOG_IA (DeepSleep AI)\n").font.size = Pt(11)
    meta.add_run("Nomenclature clinique : AASM — 5 classes\n").font.size = Pt(11)
    meta.add_run("\nDocument généré automatiquement à des fins pédagogiques.\n").italic = True
    doc.add_page_break()

    # --- 1. Résumé ---
    _add_title(doc, "1. Résumé exécutif", 1)
    doc.add_paragraph(
        "Ce projet met en œuvre une chaîne complète — des données polysomnographiques "
        "(fichiers EDF) jusqu’à une interface web — pour estimer automatiquement les stades "
        "du sommeil à partir d’un canal EOG, selon la norme AASM (cinq classes : éveil, N1, N2, N3, REM). "
        "Un réseau de neurones convolutif 1D est entraîné puis exporté au format OpenVINO (IR FP16) "
        "afin de tirer parti de l’accélération matérielle (NPU Intel AI Boost, GPU intégré ou CPU). "
        "Une application Streamlit permet le chargement des enregistrements, l’inférence, "
        "la visualisation des hypnogrammes et un rapport clinique synthétique."
    )

    # --- 2. Contexte ---
    _add_title(doc, "2. Contexte et problématique", 1)
    doc.add_paragraph(
        "La polysomnographie (PSG) enregistre plusieurs signaux physiologiques pendant une nuit. "
        "Le scoring manuel des stades par un expert est long et coûteux. "
        "L’EOG capture les mouvements oculaires caractéristiques (notamment en sommeil paradoxal, REM). "
        "L’objectif du projet est de démontrer un pipeline reproductible : prétraitement du signal, "
        "apprentissage supervisé, déploiement optimisé et outil de visualisation pour le technologue ou le chercheur."
    )

    # --- 3. Objectifs ---
    _add_title(doc, "3. Objectifs", 1)
    for t in (
        "Charger et synchroniser signaux et annotations au format EDF.",
        "Prétraiter le canal EOG (rééchantillonnage, filtrage, normalisation).",
        "Entraîner un classificateur par époques de 30 secondes (5 classes AASM).",
        "Exporter un modèle compatible NPU via OpenVINO (FP16, forme d’entrée fixe).",
        "Proposer une interface interactive (dashboard) pour l’analyse et l’export des résultats.",
    ):
        doc.add_paragraph(t, style="List Bullet")

    # --- 4. Stades AASM ---
    _add_title(doc, "4. Codification des stades (AASM)", 1)
    _add_table(
        doc,
        ["Code", "Libellé", "Description courte"],
        [
            ["0", "W (Wake)", "Éveil"],
            ["1", "N1", "Sommeil léger / transition"],
            ["2", "N2", "Sommeil léger consolidé"],
            ["3", "N3", "Sommeil profond (SWS ; N3 et N4 historiques fusionnés)"],
            ["4", "REM", "Sommeil paradoxal"],
        ],
    )
    doc.add_paragraph()

    # --- 5. Technologies ---
    _add_title(doc, "5. Technologies et bibliothèques", 1)
    doc.add_paragraph(
        "Le tableau suivant liste les principaux composants logiciels du dépôt "
        "(versions indicatives ; se référer à requirements.txt pour le déploiement du dashboard)."
    )
    _add_table(
        doc,
        ["Technologie", "Rôle dans le projet"],
        [
            ["Python 3.12 (recommandé)", "Langage principal ; voir runtime.txt pour le cloud."],
            ["Streamlit", "Interface web du laboratoire (sidebar, onglets, métriques, graphiques)."],
            ["MNE-Python 1.6", "Lecture des EDF, annotations, extraction du canal EOG."],
            ["NumPy / SciPy / Pandas", "Tableaux, statistiques, tableaux de résultats."],
            ["Matplotlib", "Hypnogrammes, camemberts d’architecture, matrices de confusion."],
            ["OpenVINO (2024.x+)", "Inférence accélérée sur NPU / GPU / CPU (modèle IR .xml + .bin)."],
            ["TensorFlow / Keras (requirements-train.txt)", "Entraînement et export des modèles .keras → OpenVINO."],
            ["scikit-learn", "Métriques et évaluation (evaluate.py)."],
        ],
    )
    doc.add_paragraph()

    # --- 6. Structure du dépôt ---
    _add_title(doc, "6. Organisation du dépôt", 1)
    _add_table(
        doc,
        ["Chemin / fichier", "Description"],
        [
            ["app/dashboard.py", "Application Streamlit : pipeline, inférence, onglets cliniques."],
            ["app/ui_theme.py", "Thème couleur centralisé (clair / sombre / système)."],
            ["src/preprocessing.py", "Rééchantillonnage 100 Hz, filtre FIR 0,5–35 Hz, z-score, clip ±3σ."],
            ["src/data_loader.py", "Parsing des annotations EDF → vecteur de labels par époque."],
            ["src/architecture.py", "Modèles CNN+Bi-LSTM (CPU) et CNN 1D pur (NPU-compatible)."],
            ["src/train.py / train_npu.py", "Scripts d’entraînement et export IR."],
            ["src/evaluate.py", "Évaluation quantitative vs vérité terrain."],
            ["models/", "Poids Keras et modèle OpenVINO IR FP16 (entrée [64, 3000, 1])."],
            ["data/raw/", "Enregistrements exemples (*_Signal.edf, *_Labels.edf)."],
            [".streamlit/config.toml", "Thème Streamlit de base."],
        ],
    )
    doc.add_paragraph()

    # --- 7. Schéma pipeline ---
    _add_title(doc, "7. Schéma du pipeline de bout en bout", 1)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        _pipeline_figure(tmp_path)
        doc.add_picture(tmp_path, width=Inches(6.2))
    finally:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        "Chaque époque correspond à 30 s à 100 Hz soit 3000 échantillons par canal. "
        "Le modèle déployé dans le dashboard attend un tenseur de forme (batch, 3000, 1) ; "
        "l’inférence est traitée par lots (batch 64) pour optimiser le débit."
    )

    # --- 8. Prétraitement ---
    _add_title(doc, "8. Prétraitement du signal EOG", 1)
    _add_table(
        doc,
        ["Étape", "Paramètres"],
        [
            ["Rééchantillonnage", "100 Hz (aligné sur l’entrée du modèle)."],
            ["Sélection du canal", "Premier canal dont le nom contient « EOG » (insensible à la casse)."],
            ["Filtrage", "Passe-bande FIR 0,5 Hz – 35 Hz (MNE)."],
            ["Normalisation", "Z-score par enregistrement ; limitation des valeurs à ±3 écarts-types."],
        ],
    )
    doc.add_paragraph()

    # --- 9. Modèles ---
    _add_title(doc, "9. Modèles d’apprentissage", 1)
    doc.add_paragraph(
        "Deux architectures sont définies dans architecture.py : une variante CNN + Bi-LSTM "
        "(meilleure expressivité temporelle mais opérations non supportées sur le NPU Intel ciblé) "
        "et une CNN 1D pure (convolutions, normalisation par lots, max-pooling, global average pooling, dense) "
        "compatible avec l’export OpenVINO et l’exécution sur NPU."
    )
    _add_table(
        doc,
        ["Critère", "CNN + Bi-LSTM", "CNN 1D (NPU)"],
        [
            ["Cible d’exécution", "CPU / GPU (LSTM)", "NPU / GPU / CPU (IR)"],
            ["Contrainte matérielle", "Ops récurrentes", "Pas de LSTM — convolutions uniquement"],
            ["Perte / optimisation", "sparse_categorical_crossentropy, Adam", "idem"],
        ],
    )
    doc.add_paragraph()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
        arch_path = tmp2.name
    try:
        _architecture_figure(arch_path)
        doc.add_picture(arch_path, width=Inches(6.0))
    finally:
        if os.path.isfile(arch_path):
            os.remove(arch_path)
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # --- 10. Dashboard ---
    _add_title(doc, "10. Fonctionnalités du dashboard", 1)
    for t in (
        "Choix du périphérique d’inférence OpenVINO (NPU, GPU, CPU, AUTO).",
        "Import EDF signal ; base locale data/raw ; hypnogramme expert optionnel.",
        "Métriques cliniques (TST, efficacité du sommeil, latences, % de stades, etc.).",
        "Onglets : hypnogramme IA (et expert), architecture du sommeil, accord IA vs expert, rapport, technique.",
        "Export CSV époque par époque.",
    ):
        doc.add_paragraph(t, style="List Bullet")

    # --- 11. Performances indicatives ---
    _add_title(doc, "11. Performances indicatives (README du projet)", 1)
    doc.add_paragraph(
        "Les chiffres ci-dessous sont donnés à titre illustratif sur une configuration type "
        "(Intel Core Ultra 5 125U) ; ils dépendent du matériel et de la charge."
    )
    _add_table(
        doc,
        ["Périphérique", "Débit approximatif (époques/s)"],
        [
            ["NPU (Intel AI Boost)", "~5 650"],
            ["GPU intégré", "~2 150"],
            ["CPU", "~1 100"],
        ],
    )
    doc.add_paragraph(
        "Précision rapportée sur un petit jeu de validation (4 patients) : environ 88,4 % "
        "pour la variante CNN 1D (voir README.md du dépôt)."
    )

    # --- 12. Limites ---
    _add_title(doc, "12. Limites et perspectives", 1)
    for t in (
        "Un seul canal EOG ne remplace pas une PSG complète pour le diagnostic clinique.",
        "Les performances varient selon la population, la qualité du signal et les pathologies.",
        "L’outil sert d’aide à la décision / démonstration pédagogique ; le scoring expert reste la référence.",
        "Pistes : multi-canaux, calibration sur plus de sujets, explicabilité (Grad-CAM), déploiement serveur.",
    ):
        doc.add_paragraph(t, style="List Bullet")

    # --- 13. Conclusion ---
    _add_title(doc, "13. Conclusion", 1)
    doc.add_paragraph(
        "Le projet Sommeil_EOG_IA illustre une chaîne moderne d’apprentissage automatique appliquée "
        "au sommeil : données standardisées (EDF / AASM), prétraitement reproductible, modèle profond "
        "déployable sur accélérateur via OpenVINO, et restitution visuelle pour l’utilisateur. "
        "Il constitue une base solide pour un exposé ou un rapport de fin d’études en traitement du signal "
        "et intelligence artificielle."
    )

    _add_title(doc, "Références logicielles", 1)
    doc.add_paragraph(
        "MNE-Python (https://mne.tools/), Streamlit (https://streamlit.io/), "
        "OpenVINO (https://docs.openvino.ai/), TensorFlow/Keras (https://www.tensorflow.org/)."
    )

    out_path = os.path.join(ROOT, OUT_NAME)
    doc.save(out_path)
    return out_path


if __name__ == "__main__":
    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    path = build_document()
    print("Document créé :", path)
