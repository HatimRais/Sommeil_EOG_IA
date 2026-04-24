"""
DeepSleep AI — Polysomnography Lab
Plateforme clinique d'analyse automatisée des stades du sommeil par IA (EOG).
Conçu pour la pratique médicale : interprétation, rapport, comparaison aux normes adultes.
"""
import os
import sys
import time
import tempfile
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import mne
import openvino as ov
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import apply_preprocessing
from src.data_loader import load_and_sync_labels, annotation_summary

# ============================================================================
# Constantes cliniques
# ============================================================================
EPOCH_DURATION = 30.0
SFREQ = 100
N_POINTS = int(EPOCH_DURATION * SFREQ)
BATCH_INFER = 64
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
STAGE_FULL = ['Wake', 'N1 (Stage 1)', 'N2 (Stage 2)', 'N3 (Slow Wave Sleep)', 'REM']

# Palette médicale (convention polysomnographie clinique)
STAGE_COLORS = {
    0: '#FFCB47',  # W   — jaune
    1: '#B8D4E3',  # N1  — bleu très clair
    2: '#6FA8DC',  # N2  — bleu moyen
    3: '#1B4965',  # N3  — bleu profond (deep sleep)
    4: '#C73E1D',  # REM — rouge (convention médicale)
}

# Valeurs normatives adulte (AASM / Carskadon-Dement)
NORMS = {
    'TST_min':       (390, 480, 'Total Sleep Time'),
    'SE_pct':        (85, 100, 'Sleep Efficiency'),
    'Latency_min':   (0, 20,  'Sleep Onset Latency'),
    'WASO_min':      (0, 30,  'Wake After Sleep Onset'),
    'REM_pct':       (20, 25, 'REM %'),
    'N1_pct':        (2, 5,   'N1 %'),
    'N2_pct':        (45, 55, 'N2 %'),
    'N3_pct':        (13, 23, 'N3 (SWS) %'),
    'REM_lat_min':   (70, 120, 'REM Latency'),
    'Awakenings':    (0, 10,  '# Awakenings'),
}

MODEL_XML = "models/sleep_model_npu.xml"
DEVICE_LABELS = {
    "NPU":  "Intel AI Boost (NPU)",
    "GPU":  "Integrated GPU",
    "CPU":  "CPU",
    "AUTO": "Auto-select",
}

# ============================================================================
# Configuration de page
# ============================================================================
st.set_page_config(
    page_title="DeepSleep AI · Polysomnography Lab",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# THEME SYSTEM — 3 modes : System / Light / Dark
# ============================================================================
THEME_LIGHT = """
    --bg:           #F4F6F9;
    --surface:      #FFFFFF;
    --surface-2:    #F9FAFB;
    --border:       #E2E8F0;
    --border-soft:  #EEF2F6;
    --text:         #0F172A;
    --text-muted:   #64748B;
    --primary:      #1B4965;
    --primary-2:    #2C7DA0;
    --primary-soft: #E6EEF4;
    --accent:       #2C7DA0;
    --success:      #15803D;
    --warning:      #B45309;
    --danger:       #B91C1C;
    --shadow:       0 1px 2px rgba(15,23,42,0.04), 0 1px 3px rgba(15,23,42,0.06);
    --shadow-hover: 0 4px 12px rgba(15,23,42,0.08);
    --sb-bg:        #EDF1F5;
    --sb-surface:   #FFFFFF;
    --sb-border:    #D1D9E2;
    --sb-text:      #0F172A;
    --sb-muted:     #64748B;
    --sb-accent:    #1B4965;
    --header-from:  #1B4965;
    --header-to:    #2C7DA0;
    --header-text:  #FFFFFF;
    --chip-norm-bg:  #ECFDF5; --chip-norm-fg:  #047857; --chip-norm-bd:  #A7F3D0;
    --chip-warn-bg:  #FFFBEB; --chip-warn-fg:  #B45309; --chip-warn-bd:  #FCD34D;
    --chip-alert-bg: #FEF2F2; --chip-alert-fg: #B91C1C; --chip-alert-bd: #FCA5A5;
    --chip-info-bg:  #EFF6FF; --chip-info-fg:  #1D4ED8; --chip-info-bd:  #BFDBFE;
"""

THEME_DARK = """
    --bg:           #0B1220;
    --surface:      #131C2E;
    --surface-2:    #182340;
    --border:       #2A3650;
    --border-soft:  #1E2A44;
    --text:         #E5EAF2;
    --text-muted:   #94A3B8;
    --primary:      #5FA8D3;
    --primary-2:    #62D2C4;
    --primary-soft: #1B2C44;
    --accent:       #62D2C4;
    --success:      #34D399;
    --warning:      #FBBF24;
    --danger:       #F87171;
    --shadow:       0 1px 2px rgba(0,0,0,0.4), 0 1px 3px rgba(0,0,0,0.3);
    --shadow-hover: 0 4px 14px rgba(0,0,0,0.5);
    --sb-bg:        #070D1A;
    --sb-surface:   #131C2E;
    --sb-border:    #2A3650;
    --sb-text:      #E5EAF2;
    --sb-muted:     #94A3B8;
    --sb-accent:    #62D2C4;
    --header-from:  #1B4965;
    --header-to:    #2C7DA0;
    --header-text:  #FFFFFF;
    --chip-norm-bg:  #0F2A1F; --chip-norm-fg:  #6EE7B7; --chip-norm-bd:  #14532D;
    --chip-warn-bg:  #2A1F0A; --chip-warn-fg:  #FCD34D; --chip-warn-bd:  #78350F;
    --chip-alert-bg: #2A0E0E; --chip-alert-fg: #FCA5A5; --chip-alert-bd: #7F1D1D;
    --chip-info-bg:  #0E1B33; --chip-info-fg:  #93C5FD; --chip-info-bd:  #1E3A8A;
"""


def _theme_vars_block(mode: str) -> str:
    if mode == "Dark":
        return f":root {{ {THEME_DARK} color-scheme: dark; }}"
    if mode == "Light":
        return f":root {{ {THEME_LIGHT} color-scheme: light; }}"
    return (
        f":root {{ {THEME_LIGHT} color-scheme: light dark; }}\n"
        f"@media (prefers-color-scheme: dark) {{ :root {{ {THEME_DARK} color-scheme: dark; }} }}"
    )


THEME_BASE_CSS = """
/* ========== Reset & global ========== */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text);
}
.stApp { background: var(--bg) !important; color: var(--text); }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }

/* Headings everywhere */
h1, h2, h3, h4, h5, h6 { color: var(--text); }
p, span, div, label, li { color: var(--text); }
a { color: var(--primary-2); }

code {
    background: var(--surface-2) !important; color: var(--text) !important;
    padding: 1px 5px; border-radius: 4px; font-size: 0.85em;
    border: 1px solid var(--border-soft);
}

/* ========== SIDEBAR ========== */
[data-testid="stSidebar"] {
    background: var(--sb-bg) !important;
    border-right: 1px solid var(--sb-border);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0.6rem; }
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--sb-text);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    color: var(--sb-text); font-weight: 600;
}
[data-testid="stSidebar"] h5 {
    color: var(--sb-accent) !important;
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.4px;
    font-weight: 700; margin: 0.7rem 0 0.4rem 0;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small {
    color: var(--sb-muted) !important; font-size: 0.78rem;
}
[data-testid="stSidebar"] hr {
    border: none; border-top: 1px solid var(--sb-border); margin: 0.8rem 0;
}
[data-testid="stSidebar"] code {
    background: var(--sb-surface) !important;
    color: var(--sb-text) !important;
    border: 1px solid var(--sb-border);
}

/* Sidebar — selectbox */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--sb-surface) !important;
    border: 1px solid var(--sb-border) !important;
    border-radius: 6px; min-height: 38px;
    color: var(--sb-text) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    border-color: var(--sb-accent) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: var(--sb-accent) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
    color: var(--sb-text) !important;
}

/* Select (sidebar) : toutes les couches (Emotion: st-emotion-cache-*, Base Web, ezrtsby2) */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div > div,
[data-testid="stSidebar"] [data-baseweb="select"] [class*="st-emotion-cache"] {
    background-color: var(--sb-surface) !important;
    background: var(--sb-surface) !important;
    color: var(--sb-text) !important;
    border-color: var(--sb-border) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [class*="st-emotion-cache"] p,
[data-testid="stSidebar"] [data-baseweb="select"] [class*="st-emotion-cache"] span,
[data-testid="stSidebar"] [data-baseweb="select"] [class*="st-emotion-cache"] label {
    color: var(--sb-text) !important;
    -webkit-text-fill-color: var(--sb-text) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [class*="st-emotion-cache"] svg {
    fill: var(--sb-accent) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] [class*="ezrtsby2"] {
    background: var(--sb-surface) !important;
    background-color: var(--sb-surface) !important;
    color: var(--sb-text) !important;
}

/* Selectbox dropdown popover (souvent en portail; aligné sidebar — selects uniquement ici) */
[data-baseweb="popover"] [role="listbox"] {
    background: var(--sb-surface) !important;
    border: 1px solid var(--sb-border) !important;
    color: var(--sb-text) !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
}
[data-baseweb="popover"] [role="option"] {
    color: var(--sb-text) !important;
    background: var(--sb-surface) !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background: var(--primary-soft) !important;
    color: var(--sb-text) !important;
}
[data-baseweb="popover"] [role="option"]:focus,
[data-baseweb="popover"] [role="option"]:active {
    background: var(--primary-soft) !important;
}
/* Couches internes Base Web (listbox) */
[data-baseweb="popover"] [role="listbox"] [class*="st-emotion-cache"] {
    background: var(--sb-surface) !important;
    color: var(--sb-text) !important;
}

/* Sidebar — radio (vertical, ex. Source) */
[data-testid="stSidebar"] .stRadio > div {
    background: var(--sb-surface) !important;
    border: 1px solid var(--sb-border) !important;
    border-radius: 6px; padding: 8px 10px; gap: 4px;
}
[data-testid="stSidebar"] .stRadio label { color: var(--sb-text) !important; padding: 4px 0; }
/* Option active (Source) : surtout thème sombre / system dark */
[data-testid="stSidebar"] .stRadio [role="radiogroup"]:not([style*="row"]) > label {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--sb-text) !important;
    border: 1px solid transparent !important;
    border-radius: 6px;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"]:not([style*="row"]) > label[data-checked="true"] {
    background: var(--primary-soft) !important;
    background-color: var(--primary-soft) !important;
    border-color: var(--sb-accent) !important;
    color: var(--sb-text) !important;
    box-shadow: inset 0 0 0 1px var(--sb-border) !important;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"]:not([style*="row"]) > label [class*="st-emotion-cache"] {
    background: transparent !important;
    color: var(--sb-text) !important;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label > div:first-child {
    background-color: transparent !important; border-color: var(--sb-muted) !important;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] input:checked ~ div,
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-checked="true"] > div:first-child {
    background-color: var(--sb-accent) !important; border-color: var(--sb-accent) !important;
}

/* Sidebar — horizontal radio (segmented : Theme System / Light / Dark) */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] {
    gap: 0 !important; padding: 3px;
    background: var(--sb-surface) !important;
    background-color: var(--sb-surface) !important;
    border: 1px solid var(--sb-border) !important;
    border-radius: 8px;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] > label {
    flex: 1; text-align: center; padding: 6px 4px; cursor: pointer;
    border-radius: 6px; transition: all 0.15s; margin: 0;
    font-size: 0.82rem; font-weight: 500;
    display: flex !important; align-items: center; justify-content: center;
    background: transparent !important;
    color: var(--sb-text) !important;
    -webkit-text-fill-color: var(--sb-text) !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] > label [class*="st-emotion-cache"] {
    background: transparent !important;
    color: inherit !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] > label:hover {
    background: var(--primary-soft) !important;
    background-color: var(--primary-soft) !important;
    color: var(--sb-text) !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] > label[data-checked="true"] {
    background: var(--sb-accent) !important;
    background-color: var(--sb-accent) !important;
    color: var(--sb-bg) !important;
    -webkit-text-fill-color: var(--sb-bg) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] > label[data-checked="true"] * {
    color: var(--sb-bg) !important;
    -webkit-text-fill-color: var(--sb-bg) !important;
    font-weight: 700;
}
/* Hide radio circles only in horizontal segmented mode */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] > label > div:first-child {
    display: none !important;
}

/* Sidebar — file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background: var(--sb-surface) !important;
    border: 1.5px dashed var(--sb-border) !important;
    border-radius: 8px; padding: 14px;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover {
    border-color: var(--sb-accent) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section *,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] p {
    color: var(--sb-text) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section small {
    color: var(--sb-muted) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: var(--sb-accent) !important; color: var(--sb-bg) !important;
    border: none !important; font-weight: 600 !important; border-radius: 5px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
    background: var(--sb-surface) !important; border: 1px solid var(--sb-border) !important;
    border-radius: 6px;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * { color: var(--sb-text) !important; }

/* Sidebar — buttons */
[data-testid="stSidebar"] .stButton > button {
    background: var(--sb-accent) !important; color: var(--sb-bg) !important;
    border: none !important; border-radius: 6px;
    padding: 0.6rem 1rem; font-weight: 700; font-size: 0.92rem;
    box-shadow: var(--shadow); transition: all 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px); box-shadow: var(--shadow-hover); filter: brightness(1.08);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--success) !important; color: #FFFFFF !important;
}

/* Sidebar — alerts */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: var(--sb-surface) !important;
    border-left: 3px solid var(--sb-accent) !important;
    border-radius: 6px;
}
[data-testid="stSidebar"] [data-testid="stAlert"] * { color: var(--sb-text) !important; }

/* ========== HEADER BAR ========== */
.med-header {
    background: linear-gradient(90deg, var(--header-from) 0%, var(--header-to) 100%);
    color: var(--header-text); padding: 1.1rem 1.6rem; border-radius: 10px;
    display: flex; justify-content: space-between; align-items: center;
    box-shadow: var(--shadow); margin-bottom: 1.2rem;
}
.med-header h1 { color: var(--header-text); margin: 0; font-size: 1.35rem; font-weight: 600; letter-spacing: -0.3px; }
.med-header .subtitle { color: rgba(255,255,255,0.78); font-size: 0.82rem; margin-top: 3px; }
.med-header .tag {
    background: rgba(255,255,255,0.15); padding: 5px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 500; margin-left: 6px;
    border: 1px solid rgba(255,255,255,0.18);
    color: var(--header-text);
}

/* ========== PATIENT CARD ========== */
.patient-card {
    background: var(--surface); border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: 8px; padding: 1rem 1.4rem; margin-bottom: 1.2rem;
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;
    box-shadow: var(--shadow);
}
.patient-card .field { display: flex; flex-direction: column; }
.patient-card .label {
    font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase;
    letter-spacing: 0.6px; font-weight: 600;
}
.patient-card .value { font-size: 0.95rem; color: var(--text); font-weight: 500; margin-top: 3px; }

/* ========== SECTION TITLES ========== */
.section-title {
    font-size: 0.92rem; font-weight: 700; color: var(--primary);
    margin: 1.6rem 0 0.9rem 0; padding-bottom: 6px;
    border-bottom: 2px solid var(--primary-2);
    letter-spacing: 0.4px; text-transform: uppercase;
}

/* ========== METRIC CARDS ========== */
div[data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 0.75rem 1rem; box-shadow: var(--shadow); transition: box-shadow 0.15s;
}
div[data-testid="stMetric"]:hover { box-shadow: var(--shadow-hover); }
div[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important; color: var(--text-muted) !important;
    text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;
}
div[data-testid="stMetricValue"] {
    color: var(--primary) !important; font-size: 1.45rem !important;
    font-weight: 700; line-height: 1.2;
}
div[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ========== CHIPS ========== */
.chip {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.4px;
    border: 1px solid transparent;
}
.chip-norm   { background: var(--chip-norm-bg);  color: var(--chip-norm-fg);  border-color: var(--chip-norm-bd); }
.chip-warn   { background: var(--chip-warn-bg);  color: var(--chip-warn-fg);  border-color: var(--chip-warn-bd); }
.chip-alert  { background: var(--chip-alert-bg); color: var(--chip-alert-fg); border-color: var(--chip-alert-bd); }
.chip-info   { background: var(--chip-info-bg);  color: var(--chip-info-fg);  border-color: var(--chip-info-bd); }

/* ========== TABS ========== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; border-bottom: 1px solid var(--border); background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; padding: 10px 18px; color: var(--text-muted);
    font-weight: 500; border-bottom: 2px solid transparent; transition: all 0.15s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--primary); background: var(--primary-soft);
}
.stTabs [aria-selected="true"] {
    color: var(--primary) !important; border-bottom: 2px solid var(--primary) !important;
    font-weight: 600;
}

/* ========== MAIN BUTTONS ========== */
.stButton > button, .stDownloadButton > button {
    background: var(--primary); color: #FFFFFF; border: none; border-radius: 6px;
    padding: 0.55rem 1.3rem; font-weight: 600; font-size: 0.88rem;
    transition: all 0.15s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    filter: brightness(1.1); transform: translateY(-1px); box-shadow: var(--shadow-hover);
}

/* ========== DATA TABLES & DATAFRAMES ========== */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border); border-radius: 6px; overflow: hidden;
    background: var(--surface);
}

/* ========== ALERTS / INFO ========== */
[data-testid="stAlert"] {
    border-radius: 6px; border-left-width: 4px;
}

/* ========== PROGRESS BAR ========== */
.stProgress > div > div > div > div { background: var(--primary-2); }

/* ========== DISCLAIMER ========== */
.disclaimer {
    background: var(--chip-warn-bg); border-left: 4px solid var(--warning);
    border-radius: 4px; padding: 10px 14px;
    font-size: 0.8rem; color: var(--chip-warn-fg); margin-top: 1.5rem;
}

/* ========== EXPANDER ========== */
[data-testid="stExpander"] {
    background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
}
[data-testid="stExpander"] summary { color: var(--text); }

/* ========== JSON / CODE BLOCKS ========== */
[data-testid="stJson"], [data-testid="stCodeBlock"], pre {
    background: var(--surface) !important; border: 1px solid var(--border);
    border-radius: 6px; color: var(--text);
}
[data-testid="stJson"] *, [data-testid="stCodeBlock"] *, pre code {
    color: var(--text) !important;
}
[data-testid="stJson"] .key { color: var(--primary-2) !important; }
[data-testid="stJson"] .string { color: var(--success) !important; }
[data-testid="stJson"] .number { color: var(--warning) !important; }

/* ========== DATAFRAMES (native) ========== */
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
    background: var(--surface) !important;
}
[data-testid="stDataFrame"] table, [data-testid="stDataFrame"] thead tr th,
[data-testid="stDataFrame"] tbody tr td {
    background: var(--surface) !important; color: var(--text) !important;
    border-color: var(--border) !important;
}
[data-testid="stDataFrame"] thead tr th {
    background: var(--surface-2) !important; font-weight: 600;
}

/* ========== TEXT INPUT / MAIN SELECTBOX ========== */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stTextInput > div > div {
    background: var(--surface); border-color: var(--border);
}

/* Helper utility classes for inline annotations */
.muted { color: var(--text-muted); }
.norm-line { font-size: 0.78rem; color: var(--text-muted); }

/* Clinical interpretation boxes */
.interp-box {
    background: var(--surface-2); border-left: 3px solid var(--primary);
    border-radius: 4px; padding: 10px 14px; margin-bottom: 6px;
    color: var(--text);
}
.interp-box.norm  { border-left-color: var(--success); }
.interp-box.warn  { border-left-color: var(--warning); }
.interp-box.alert { border-left-color: var(--danger); }

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }
"""

# Responsive: phones & small tablets (Streamlit main area uses stHorizontalBlock / stColumn)
THEME_RESPONSIVE_CSS = """
/* ========== TABLET (≤ 900px) — softer layout ========== */
@media (max-width: 900px) {
  .block-container { max-width: 100% !important; padding-left: 1rem; padding-right: 1rem; }
  .med-header { flex-wrap: wrap; gap: 0.75rem; }
  .med-header h1 { font-size: 1.15rem; }
  .stTabs [data-baseweb="tab-list"] { flex-wrap: nowrap; overflow-x: auto; -webkit-overflow-scrolling: touch; gap: 2px; }
  .stTabs [data-baseweb="tab"] { padding: 8px 12px; font-size: 0.8rem; white-space: nowrap; }
}

/* Main content only (excludes sidebar) — stHorizontalBlock wraps each st.columns() row */
/* Streamlit: section.main in app view, or [data-testid="stMain"] in newer versions */
/* ========== PHONE (≤ 768px) — stack columns, touch-friendly ========== */
@media (max-width: 768px) {
  .stApp { padding-left: env(safe-area-inset-left, 0); padding-right: env(safe-area-inset-right, 0); }
  .block-container { padding-top: 0.6rem; padding-bottom: 1.2rem; padding-left: 0.65rem; padding-right: 0.65rem; }

  [data-testid="stAppViewContainer"] .main, [data-testid="stMain"] {
    padding-bottom: max(0.5rem, env(safe-area-inset-bottom, 0));
  }

  /* Stack Streamlit column rows (metrics, Welcome columns, tab rows) — main only */
  section.main [data-testid="stHorizontalBlock"],
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    gap: 0.75rem !important;
  }
  section.main [data-testid="stHorizontalBlock"] [data-testid="column"],
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] [data-testid="column"] {
    width: 100% !important;
    min-width: 0 !important;
    flex: 1 1 auto !important;
  }

  .med-header {
    flex-direction: column; align-items: flex-start;
    padding: 0.9rem 1rem; border-radius: 8px;
  }
  .med-header h1 { font-size: 1.05rem; line-height: 1.3; }
  .med-header .subtitle { font-size: 0.75rem; }
  .med-header .tag { font-size: 0.7rem; padding: 4px 8px; margin: 0 6px 0 0; }

  .patient-card {
    grid-template-columns: 1fr; gap: 0.75rem; padding: 0.85rem 1rem;
  }
  .section-title { font-size: 0.85rem; margin: 1.1rem 0 0.65rem; }

  div[data-testid="stMetric"] { padding: 0.65rem 0.75rem; }
  div[data-testid="stMetricValue"] { font-size: 1.25rem !important; }

  .stButton > button, .stDownloadButton > button,
  [data-testid="stSidebar"] .stButton > button {
    min-height: 44px; padding: 0.65rem 1rem; font-size: 0.9rem;
  }
  [data-testid="stSidebar"] [data-baseweb="select"] > div { min-height: 44px; }
  [data-testid="stFileUploader"] section { min-height: 100px; }

  [data-baseweb="popover"] { max-width: min(100vw, 100%); }

  [data-testid="stDataFrame"] { max-width: 100%; overflow: auto; -webkit-overflow-scrolling: touch; }

  .stButton > button, .stDownloadButton > button { -webkit-tap-highlight-color: transparent; }
}

/* ========== NARROW PHONE (≤ 480px) — single column, compact type ========== */
@media (max-width: 480px) {
  .med-header h1 { font-size: 0.98rem; }
  .stTabs [data-baseweb="tab"] { padding: 6px 8px; font-size: 0.72rem; }
  [data-testid="stSidebar"] [data-baseweb="select"] > div, [data-testid="stSidebar"] [data-baseweb="input"] > div { font-size: 0.88rem; }
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 0.95rem; }

  [data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] {
    flex-wrap: wrap;
  }
  [data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] > label {
    min-width: 0; font-size: 0.7rem; padding: 5px 2px;
  }

  .disclaimer, .interp-box { font-size: 0.75rem; padding: 8px 10px; }
}

/* iOS: avoid automatic zoom on focus in fields (16px+ font) */
@media (max-width: 768px) {
  [data-baseweb="input"] input,
  [data-baseweb="select"] input,
  input[type="text"] {
    font-size: 16px !important;
  }
}
"""

# Dernière couche: Streamlit charge son thème (config + emotion) après nos règles.
# Ciblage fort + !important + réinjection en fin de script (theme_style_last).
THEME_OVERRIDES_CSS = """
/* ——— App shell (fond d’écran, zone principale) ——— */
.stApp, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  background-image: none !important;
}
section.main, [data-testid="stMain"] {
  background: transparent !important;
  color: var(--text) !important;
}
div[data-testid="block-container"] {
  background: transparent !important;
  color: var(--text) !important;
}

/* ——— Markdown (zone principale + onglets) — Streamlit met souvent couleur thème clair sur <p> ——— */
section.main [data-testid="stMarkdownContainer"] p,
section.main [data-testid="stMarkdownContainer"] li,
section.main [data-testid="stMarkdownContainer"] span,
section.main [data-testid="stMarkdownContainer"] strong,
section.main [data-testid="stMarkdownContainer"] b,
section.main [data-testid="stMarkdownContainer"] em,
[role="tabpanel"] [data-testid="stMarkdownContainer"] p,
[role="tabpanel"] [data-testid="stMarkdownContainer"] li,
[role="tabpanel"] [data-testid="stMarkdownContainer"] span,
[role="tabpanel"] [data-testid="stMarkdownContainer"] strong,
[role="tabpanel"] [data-testid="stMarkdownContainer"] b {
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
}
section.main a, [role="tabpanel"] [data-testid="stMarkdownContainer"] a {
  color: var(--primary-2) !important;
}

/* ——— Sidebar: tout le texte (y compris gras) ——— */
[data-testid="stSidebar"] {
  background: var(--sb-bg) !important;
  background-image: none !important;
  color: var(--sb-text) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] b,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
  color: var(--sb-text) !important;
  -webkit-text-fill-color: var(--sb-text) !important;
}
[data-testid="stSidebar"] h5, [data-testid="stSidebar"] [data-testid="stHeader"] h1,
[data-testid="stSidebar"] [data-testid="stHeader"] h2, [data-testid="stSidebar"] [data-testid="stHeader"] h3 {
  color: var(--sb-accent) !important;
  -webkit-text-fill-color: var(--sb-accent) !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li { color: var(--sb-text) !important; }

/* Select / inputs sidebar (Base Web + Emotion) */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div > div,
[data-testid="stSidebar"] [data-baseweb="select"] [class*="st-emotion-cache"],
[data-testid="stSidebar"] [data-baseweb="select"] [class*="ezrtsby2"] {
  background-color: var(--sb-surface) !important;
  background: var(--sb-surface) !important;
  border-color: var(--sb-border) !important;
  color: var(--sb-text) !important;
  box-shadow: none !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-baseweb="select"] span[role="combobox"] {
  color: var(--sb-text) !important;
  -webkit-text-fill-color: var(--sb-text) !important;
  caret-color: var(--sb-text) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: var(--sb-accent) !important; }

/* Liste déroulante portail (Hardware) : aligner sb_* — System dark / Dark */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="listbox"] [class*="st-emotion-cache"] {
  background: var(--sb-surface) !important;
  background-color: var(--sb-surface) !important;
  color: var(--sb-text) !important;
  border-color: var(--sb-border) !important;
}
[data-baseweb="popover"] [role="option"] {
  background: var(--sb-surface) !important;
  color: var(--sb-text) !important;
  -webkit-text-fill-color: var(--sb-text) !important;
}
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] [role="option"]:hover {
  background: var(--primary-soft) !important;
  color: var(--sb-text) !important;
}

/* Radio Theme (ligne) + Source (colonne) */
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] > label[data-checked="true"] {
  background: var(--sb-accent) !important;
  background-color: var(--sb-accent) !important;
  color: var(--sb-bg) !important;
  -webkit-text-fill-color: var(--sb-bg) !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] > label[data-checked="true"] * {
  color: var(--sb-bg) !important;
  -webkit-text-fill-color: var(--sb-bg) !important;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"]:not([style*="row"]) > label[data-checked="true"] {
  background: var(--primary-soft) !important;
  background-color: var(--primary-soft) !important;
  border-color: var(--sb-accent) !important;
  color: var(--sb-text) !important;
}

/* Radio: libellés */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
[data-testid="stSidebar"] [data-baseweb="radio"] label,
[data-testid="stSidebar"] .stRadio label { color: var(--sb-text) !important; }

/* File uploader (texte seulement, pas le bouton browse) */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section p,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section small,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section [data-testid="stMarkdownContainer"] * {
  color: var(--sb-text) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
  color: var(--sb-bg) !important;
  -webkit-text-fill-color: var(--sb-bg) !important;
}

/* Métriques (labels/valeurs parfois noirs) */
div[data-testid="stMetric"] {
  background-color: var(--surface) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
}
div[data-testid="stMetricLabel"] > div, div[data-testid="stMetricLabel"] {
  color: var(--text-muted) !important;
  -webkit-text-fill-color: var(--text-muted) !important;
}
div[data-testid="stMetricValue"] > div, div[data-testid="stMetricValue"] {
  color: var(--primary) !important;
  -webkit-text-fill-color: var(--primary) !important;
}

/* Onglets */
.stTabs [data-baseweb="tab"] { color: var(--text-muted) !important; }
.stTabs [aria-selected="true"] { color: var(--primary) !important; -webkit-text-fill-color: var(--primary) !important; }

/* Légende hypnogram (HTML) */
span.med-legend-txt { color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; }
"""


def inject_theme(mode: str):
    """Inject CSS for the chosen theme mode (System | Light | Dark)."""
    st.markdown(
        f"<style>{_theme_vars_block(mode)}\n{THEME_BASE_CSS}\n{THEME_RESPONSIVE_CSS}\n{THEME_OVERRIDES_CSS}</style>",
        unsafe_allow_html=True,
    )


def theme_style_last():
    """
    Répète les règles fortes en fin de page pour gagner sur le thème natif Streamlit
    (ordre CSS + même spécificité).
    """
    st.markdown(
        f"<style data-dps-theme-overrides='1'>{THEME_OVERRIDES_CSS}</style>",
        unsafe_allow_html=True,
    )


# Initialise + inject theme BEFORE the rest of the page renders
if "theme" not in st.session_state:
    st.session_state["theme"] = "System"
inject_theme(st.session_state["theme"])


# ============================================================================
# Matplotlib palette synchronised with the active theme
# ============================================================================
PLOT_LIGHT = dict(
    fig_bg="#FFFFFF", axes_bg="#FAFBFC",
    text="#1F2937", muted="#6B7280",
    spine="#D1D5DB", grid="#E5E7EB",
    primary="#1B4965", primary_2="#5FA8D3", accent="#2C7DA0",
    pie_edge="#FFFFFF",
    cm_text_dark="#1F2937", cm_text_light="#FFFFFF",
    stage_colors={0:'#FFCB47', 1:'#B8D4E3', 2:'#6FA8DC', 3:'#1B4965', 4:'#C73E1D'},
    cmap='Blues',
)
PLOT_DARK = dict(
    fig_bg="#131C2E", axes_bg="#0B1220",
    text="#E5EAF2", muted="#94A3B8",
    spine="#2A3650", grid="#1E2A44",
    primary="#5FA8D3", primary_2="#82C0E1", accent="#62D2C4",
    pie_edge="#131C2E",
    cm_text_dark="#0B1220", cm_text_light="#FFFFFF",
    stage_colors={0:'#FCD34D', 1:'#B8D4E3', 2:'#5FA8D3', 3:'#62D2C4', 4:'#F87171'},
    cmap='Blues',
)


def get_plot_palette() -> dict:
    """Return matplotlib colors matching the active Streamlit theme."""
    mode = st.session_state.get("theme", "System")
    return PLOT_DARK if mode == "Dark" else PLOT_LIGHT


# ============================================================================
# Helpers
# ============================================================================
@st.cache_resource(show_spinner=False)
def list_devices():
    try:
        return list(ov.Core().available_devices)
    except Exception:
        return []


@st.cache_resource(show_spinner="Initializing inference engine…")
def load_engine(device: str):
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
            base = exec_dev.split('.')[0]
            hw = core.get_property(base, "FULL_DEVICE_NAME")
        except Exception:
            hw = exec_dev
        return net, exec_dev, hw
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}", None


def save_upload_to_temp(uploaded) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
    try:
        tmp.write(uploaded.getvalue())
    finally:
        tmp.close()
    return tmp.name


def infer_batched(net, X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    pad = (-n) % BATCH_INFER
    if pad:
        X = np.concatenate([X, np.zeros((pad,) + X.shape[1:], dtype=X.dtype)], axis=0)
    outs, key = [], net.output(0)
    for i in range(0, X.shape[0], BATCH_INFER):
        outs.append(net(X[i:i + BATCH_INFER])[key])
    return np.concatenate(outs, axis=0)[:n]


def anonymous_id(path: str) -> str:
    return "PSG-" + hashlib.md5(path.encode()).hexdigest()[:8].upper()


def fmt_hms(minutes: float) -> str:
    if not np.isfinite(minutes): return "—"
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}h {m:02d}min"


def clinical_status(value, lo, hi, lower_better=False):
    """Retourne (chip_html, statut) selon plage normative."""
    if value is None or not np.isfinite(value):
        return "<span class='chip chip-info'>—</span>", "n/a"
    if lower_better:
        if value <= hi:
            return "<span class='chip chip-norm'>NORMAL</span>", "normal"
        elif value <= hi * 1.5:
            return "<span class='chip chip-warn'>BORDERLINE</span>", "warn"
        else:
            return "<span class='chip chip-alert'>ELEVATED</span>", "alert"
    if lo <= value <= hi:
        return "<span class='chip chip-norm'>NORMAL</span>", "normal"
    if value < lo:
        if value >= lo * 0.85:
            return "<span class='chip chip-warn'>LOW</span>", "warn"
        return "<span class='chip chip-alert'>LOW</span>", "alert"
    if value <= hi * 1.15:
        return "<span class='chip chip-warn'>HIGH</span>", "warn"
    return "<span class='chip chip-alert'>HIGH</span>", "alert"


def detect_sleep_cycles(preds: np.ndarray) -> int:
    """Compte les cycles REM (transition vers REM après une période de sommeil)."""
    cycles = 0
    in_rem = False
    rem_min_epochs = 2  # ≥ 1 minute de REM continu
    i = 0
    while i < len(preds):
        if preds[i] == 4:
            run = 0
            while i < len(preds) and preds[i] == 4:
                run += 1; i += 1
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
        rem_lat = rem_idx[0] * epoch_dur / 60.0 if len(rem_idx) else float('nan')
    else:
        sol = float('nan'); waso = 0; rem_lat = float('nan')

    counts = {s: int((preds == i).sum()) for i, s in enumerate(STAGES)}
    sleep_total = max(1, sleep.sum())
    pct = {
        'W_pct':   counts['W']   / max(1, n) * 100,
        'N1_pct':  counts['N1']  / sleep_total * 100,
        'N2_pct':  counts['N2']  / sleep_total * 100,
        'N3_pct':  counts['N3']  / sleep_total * 100,
        'REM_pct': counts['REM'] / sleep_total * 100,
    }
    awakenings = int(np.sum((preds[:-1] != 0) & (preds[1:] == 0)))
    transitions = int((np.diff(preds) != 0).sum())
    fragmentation = transitions / max(1, sleep.sum()) * 100  # arousals/sleep epoch

    return {
        'TIB_min': tib, 'TST_min': tst, 'SE_pct': se,
        'Latency_min': sol, 'WASO_min': waso, 'REM_lat_min': rem_lat,
        'Awakenings': awakenings, 'Transitions': transitions,
        'Fragmentation_idx': fragmentation,
        'Cycles': detect_sleep_cycles(preds),
        'Counts': counts, **pct,
    }


def plot_hypnogram(ax, stages, title='Hypnogram', highlight_rem=True, palette=None):
    p = palette or get_plot_palette()
    n = len(stages)
    x = np.arange(n) * EPOCH_DURATION / 3600.0
    rem_mask = (stages == 4).astype(float) if not np.any(np.isnan(stages)) else None

    ax.step(x, stages, where='post', color=p['primary'], linewidth=1.2)
    ax.fill_between(x, stages, step='post', color=p['primary_2'], alpha=0.18)

    if highlight_rem and rem_mask is not None and np.any(rem_mask):
        rem_x = np.where(rem_mask > 0)[0] * EPOCH_DURATION / 3600.0
        for rx in rem_x:
            ax.axvspan(rx, rx + EPOCH_DURATION / 3600.0,
                       color=p['stage_colors'][4], alpha=0.18, zorder=0)

    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(STAGES, fontsize=9, color=p['text'])
    ax.set_ylim(-0.5, 4.5)
    ax.invert_yaxis()
    ax.set_xlabel('Time from recording onset (hours)', fontsize=9, color=p['muted'])
    ax.set_title(title, fontsize=10.5, color=p['primary'], fontweight='600', loc='left', pad=8)
    ax.grid(True, axis='both', alpha=0.35, linestyle='--', linewidth=0.5, color=p['grid'])
    ax.tick_params(colors=p['muted'], labelsize=8)
    for s in ax.spines.values():
        s.set_color(p['spine'])
    ax.set_facecolor(p['axes_bg'])


def plot_architecture_pie(ax, counts: dict, palette=None):
    """Pie chart de l'architecture du sommeil (% du TST)."""
    p = palette or get_plot_palette()
    labels = ['N1', 'N2', 'N3', 'REM']
    values = [counts[l] for l in labels]
    colors = [p['stage_colors'][i] for i in range(1, 5)]
    if sum(values) == 0:
        ax.text(0.5, 0.5, 'No sleep detected', ha='center', va='center', color=p['muted'])
        ax.axis('off'); return
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, wedgeprops=dict(edgecolor=p['pie_edge'], linewidth=2),
        textprops=dict(fontsize=9, color=p['text']),
    )
    for at in autotexts:
        at.set_color('#FFFFFF'); at.set_fontweight('700'); at.set_fontsize(8.5)
    ax.set_title('Sleep Architecture (% of TST)', fontsize=10.5,
                 color=p['primary'], fontweight='600', loc='left', pad=8)


def plot_confusion(ax, y_true, y_pred, palette=None):
    p = palette or get_plot_palette()
    valid = y_true != -1
    yt, yp = y_true[valid], y_pred[valid]
    cm = np.zeros((5, 5), dtype=int)
    for t, p_ in zip(yt, yp):
        cm[t, p_] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    ax.imshow(cm_norm, cmap=p['cmap'], vmin=0, vmax=100)
    for i in range(5):
        for j in range(5):
            color = p['cm_text_light'] if cm_norm[i, j] > 55 else p['cm_text_dark']
            ax.text(j, i, f"{cm_norm[i, j]:.0f}%", ha='center', va='center',
                    color=color, fontsize=9, fontweight='700')
            ax.text(j, i + 0.32, f"n={cm[i, j]}", ha='center', va='center',
                    color=color, fontsize=6.5, alpha=0.85)
    ax.set_xticks(range(5)); ax.set_xticklabels(STAGES, fontsize=8.5, color=p['text'])
    ax.set_yticks(range(5)); ax.set_yticklabels(STAGES, fontsize=8.5, color=p['text'])
    ax.set_xlabel("AI prediction", fontsize=9, color=p['muted'])
    ax.set_ylabel("Expert (ground truth)", fontsize=9, color=p['muted'])
    ax.set_title("Confusion Matrix (row %)", fontsize=10.5,
                 color=p['primary'], fontweight='600', loc='left', pad=8)
    ax.tick_params(colors=p['muted'])
    for s in ax.spines.values(): s.set_color(p['spine'])


def per_class_metrics(y_true, y_pred) -> pd.DataFrame:
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
        rows.append({
            "Stage": name,
            "Sensitivity (Recall)": f"{rec*100:.1f}%",
            "Precision (PPV)": f"{prec*100:.1f}%",
            "F1-score": f"{f1*100:.1f}%",
            "Support (epochs)": sup,
        })
    return pd.DataFrame(rows)


def cohens_kappa(y_true, y_pred) -> float:
    """Cohen's kappa entre IA et expert (gold standard pour PSG)."""
    valid = y_true != -1
    yt, yp = y_true[valid], y_pred[valid]
    n = len(yt)
    if n == 0: return float('nan')
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(yt, yp):
        cm[t, p] += 1
    po = np.trace(cm) / n
    pe = sum(cm[i].sum() * cm[:, i].sum() for i in range(5)) / (n ** 2)
    if pe == 1: return 1.0
    return (po - pe) / (1 - pe)


def make_report_csv(preds, y_true=None) -> bytes:
    df = pd.DataFrame({
        "epoch": np.arange(len(preds)),
        "time_hms": [f"{int(i*EPOCH_DURATION//3600):02d}:"
                     f"{int((i*EPOCH_DURATION%3600)//60):02d}:"
                     f"{int(i*EPOCH_DURATION%60):02d}" for i in range(len(preds))],
        "stage_AI": [STAGES[p] for p in preds],
    })
    if y_true is not None:
        df["stage_expert"] = [STAGES[t] if t >= 0 else "?" for t in y_true]
        df["agreement"] = [
            "OK" if (t >= 0 and p == t) else ("MISMATCH" if t >= 0 else "")
            for p, t in zip(preds, y_true)
        ]
    return df.to_csv(index=False).encode("utf-8")


# ============================================================================
# Sidebar
# ============================================================================
available = list_devices()

with st.sidebar:
    st.markdown("## ⚕️ DeepSleep AI")
    st.markdown("**Polysomnography Lab**")
    st.caption("v2.0 · OpenVINO inference engine")
    st.markdown("---")

    # Theme switcher (System / Light / Dark)
    st.markdown("##### Appearance")
    theme_options = ["System", "Light", "Dark"]
    new_theme = st.radio(
        "Theme",
        theme_options,
        index=theme_options.index(st.session_state.get("theme", "System")),
        horizontal=True,
        label_visibility="collapsed",
        key="theme_selector",
    )
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    st.markdown("---")

    # Device
    st.markdown("##### Hardware acceleration")
    if not available:
        st.error("OpenVINO: no device detected.")
        st.stop()
    options = [d for d in ["NPU", "GPU", "CPU"] if d in available]
    options.append("AUTO")
    device_choice = st.selectbox(
        "Device", options=options, index=0,
        format_func=lambda d: DEVICE_LABELS.get(d, d),
        label_visibility="collapsed",
    )
    npu_net, exec_dev, hw_name = load_engine(device_choice)
    if npu_net is None:
        st.error(exec_dev); st.stop()
    st.caption(f"`{exec_dev}` · {hw_name}")

    st.markdown("---")

    # Source
    st.markdown("##### Patient data")
    src = st.radio(
        "Source",
        ["Upload EDF files", "Internal database"],
        label_visibility="collapsed",
    )
    up_sig, up_lab, local_sig, local_lab = None, None, None, None

    if src == "Upload EDF files":
        up_sig = st.file_uploader("PSG signal (.edf)", type=["edf"], key="sig")
        up_lab = st.file_uploader("Expert hypnogram (.edf) — optional",
                                  type=["edf"], key="lab")
    else:
        data_dir = "data/raw"
        if os.path.isdir(data_dir):
            sigs = sorted([f for f in os.listdir(data_dir) if "Signal.edf" in f])
            choice = st.selectbox("Patient record", ["—"] + sigs)
            if choice != "—":
                local_sig = os.path.join(data_dir, choice)
                cand = os.path.join(data_dir, choice.replace("Signal.edf", "Labels.edf"))
                if os.path.exists(cand):
                    local_lab = cand
                    st.caption(f"✓ Expert hypnogram detected")
        else:
            st.warning("`data/raw/` not found.")

    st.markdown("---")
    run = st.button("▶ Analyze recording", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("**Model**: 1D-CNN (455k params, FP16)")
    st.caption("**Standard**: AASM 5-class scoring")
    st.caption("**Epoch length**: 30 s @ 100 Hz")


# ============================================================================
# Header bar
# ============================================================================
st.markdown(f"""
<div class="med-header">
    <div>
        <h1>⚕️ DeepSleep AI · Polysomnography Lab</h1>
        <div class="subtitle">Automated EOG-based sleep staging · AASM 5-class scoring</div>
    </div>
    <div>
        <span class="tag">{datetime.now().strftime('%Y-%m-%d')}</span>
        <span class="tag">Engine: {exec_dev}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# Landing state
# ============================================================================
if not run:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("""
        ### Welcome
        This platform performs automated polysomnographic scoring from a single **EOG (electro-oculogram)** channel,
        applying the **AASM 5-class** standard (W · N1 · N2 · N3 · REM).

        **Workflow**
        1. Select hardware accelerator (sidebar)
        2. Upload a `.edf` PSG recording, or pick a record from the internal database
        3. Optionally upload the expert hypnogram for validation
        4. Click **Analyze recording**

        Quantitative results, confidence statistics, sleep architecture, hypnogram and (when available)
        agreement metrics with the expert score will be displayed.
        """)
    with col2:
        st.markdown("##### Reference values (adult)")
        for k, (lo, hi, label) in NORMS.items():
            unit = "min" if "min" in k or "Latency" in k else ("%" if "pct" in k else "")
            st.markdown(
                f"<small><b>{label}</b><br>"
                f"<span class='muted'>{lo}–{hi} {unit}</span></small>",
                unsafe_allow_html=True,
            )
    with col3:
        st.markdown("##### About the model")
        st.markdown("""
        <small>
        <b>Architecture</b>: 1D Convolutional Neural Network (CNN)<br>
        <b>Training data</b>: Multi-subject EOG recordings, AASM scored<br>
        <b>Mean accuracy</b>: ~88% vs human expert<br>
        <b>Cohen's κ</b>: ~0.78 (substantial agreement)<br>
        <b>Inference</b>: OpenVINO · FP16 · NPU/GPU/CPU
        </small>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div class='disclaimer'>
    <b>⚠ Clinical disclaimer.</b> This tool is intended as a <b>decision-support assistant</b>
    for sleep technologists and physicians. It does not replace expert visual scoring of polysomnography.
    All AI predictions should be verified by qualified personnel.
    </div>
    """, unsafe_allow_html=True)
    theme_style_last()
    st.stop()


# ============================================================================
# Pipeline
# ============================================================================
sig_path = save_upload_to_temp(up_sig) if up_sig else local_sig
lab_path = save_upload_to_temp(up_lab) if up_lab else local_lab
if not sig_path:
    st.warning("No signal file provided."); st.stop()

prog = st.progress(0, text="Loading EDF…")
try:
    prog.progress(15, "Reading EDF header…")
    raw = mne.io.read_raw_edf(sig_path, preload=True, verbose=False)
    sfreq_orig = raw.info['sfreq']; duration_min = raw.times[-1] / 60.0
    n_channels = len(raw.ch_names); meas_date = raw.info.get('meas_date')

    prog.progress(40, "Preprocessing (resample · filter · z-score)…")
    data, eog_ch = apply_preprocessing(raw)

    prog.progress(60, "Epoching (30-s windows)…")
    n_epochs = len(data) // N_POINTS
    if n_epochs == 0:
        prog.empty(); st.error("Recording too short (no full 30-s epoch)."); st.stop()
    X = data[:n_epochs * N_POINTS].reshape(n_epochs, N_POINTS, 1).astype(np.float32)

    prog.progress(80, f"Inference on {exec_dev}…")
    t0 = time.time()
    raw_out = infer_batched(npu_net, X)
    preds = np.argmax(raw_out, axis=1)
    confs = np.max(raw_out, axis=1)
    inf_t = time.time() - t0

    y_true = None
    if lab_path:
        prog.progress(95, "Synchronizing expert hypnogram…")
        y_true = load_and_sync_labels(lab_path, n_epochs)

    prog.progress(100, "Done"); time.sleep(0.2); prog.empty()
except Exception as e:
    prog.empty(); st.error(f"Pipeline error: {e}"); st.exception(e); st.stop()

report = compute_clinical_report(preds)
overall_acc = None; kappa = None
if y_true is not None and not np.all(y_true == -1):
    valid = y_true != -1
    overall_acc = (preds[valid] == y_true[valid]).mean() * 100
    kappa = cohens_kappa(y_true, preds)


# ============================================================================
# Patient card
# ============================================================================
patient_id = anonymous_id(sig_path)
rec_date = meas_date.strftime('%Y-%m-%d %H:%M') if meas_date else 'unknown'

st.markdown(f"""
<div class="patient-card">
    <div class="field"><span class="label">Record ID</span><span class="value">{patient_id}</span></div>
    <div class="field"><span class="label">Recording date</span><span class="value">{rec_date}</span></div>
    <div class="field"><span class="label">Duration</span><span class="value">{fmt_hms(duration_min)}</span></div>
    <div class="field"><span class="label">Source file</span><span class="value">{os.path.basename(sig_path)}</span></div>
    <div class="field"><span class="label">EOG channel</span><span class="value">{eog_ch}</span></div>
    <div class="field"><span class="label">Sampling</span><span class="value">{sfreq_orig:.0f} → {SFREQ} Hz</span></div>
    <div class="field"><span class="label">Epochs analyzed</span><span class="value">{n_epochs}</span></div>
    <div class="field"><span class="label">Inference</span><span class="value">{inf_t*1000:.0f} ms</span></div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# Clinical metrics
# ============================================================================
st.markdown('<div class="section-title">Clinical Sleep Metrics</div>', unsafe_allow_html=True)

def metric_with_status(col, label, value, value_str, norm_key, lower_better=False):
    lo, hi, _ = NORMS[norm_key]
    chip, _ = clinical_status(value, lo, hi, lower_better=lower_better)
    col.metric(label, value_str)
    col.markdown(f"{chip} <small class='muted'>ref: {lo}–{hi}</small>",
                 unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
metric_with_status(c1, "Total Sleep Time", report['TST_min'], fmt_hms(report['TST_min']), 'TST_min')
metric_with_status(c2, "Sleep Efficiency", report['SE_pct'], f"{report['SE_pct']:.1f} %", 'SE_pct')
metric_with_status(c3, "Sleep Latency",   report['Latency_min'], fmt_hms(report['Latency_min']), 'Latency_min', lower_better=True)
metric_with_status(c4, "WASO",            report['WASO_min'], fmt_hms(report['WASO_min']), 'WASO_min', lower_better=True)
metric_with_status(c5, "REM Latency",     report['REM_lat_min'], fmt_hms(report['REM_lat_min']), 'REM_lat_min')

c6, c7, c8, c9, c10 = st.columns(5)
metric_with_status(c6, "REM %",  report['REM_pct'], f"{report['REM_pct']:.1f} %", 'REM_pct')
metric_with_status(c7, "N3 (SWS) %", report['N3_pct'], f"{report['N3_pct']:.1f} %", 'N3_pct')
metric_with_status(c8, "N2 %",   report['N2_pct'], f"{report['N2_pct']:.1f} %", 'N2_pct')
metric_with_status(c9, "Awakenings", report['Awakenings'], str(report['Awakenings']), 'Awakenings', lower_better=True)
c10.metric("Sleep Cycles", str(report['Cycles']))
c10.markdown("<span class='chip chip-info'>typical: 4–6</span>", unsafe_allow_html=True)


# ============================================================================
# Tabs
# ============================================================================
tab_hypno, tab_arch, tab_perf, tab_report, tab_tech = st.tabs([
    "📈 Hypnogram", "🥧 Sleep Architecture", "✓ AI vs Expert", "📋 Clinical Report", "⚙ Technical",
])

# --- Hypnogram ---
with tab_hypno:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titleweight': '600'})
    palette = get_plot_palette()
    n_plots = 2 if (y_true is not None and not np.all(y_true == -1)) else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.0 * n_plots), sharex=True)
    fig.patch.set_facecolor(palette['fig_bg'])
    if n_plots == 1: axes = [axes]
    plot_hypnogram(axes[0], preds.astype(float), title='AI-predicted hypnogram', palette=palette)
    if n_plots == 2:
        y_show = np.where(y_true == -1, np.nan, y_true).astype(float)
        plot_hypnogram(axes[1], y_show, title='Expert-scored hypnogram (ground truth)', palette=palette)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Legend (uses theme stage colors)
    legend_html = "<div style='display:flex; gap:14px; flex-wrap:wrap; margin-top:6px'>"
    for i, name in enumerate(STAGES):
        legend_html += (
            f"<div style='display:flex; align-items:center; gap:6px'>"
            f"<span style='width:14px; height:14px; background:{palette['stage_colors'][i]}; "
            f"border-radius:3px; display:inline-block'></span>"
            f"<span class='med-legend-txt'>{STAGE_FULL[i]}</span></div>"
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    if y_true is not None and np.all(y_true == -1):
        st.warning("Expert hypnogram could not be parsed.")
        descs, _ = annotation_summary(lab_path)
        st.code("\n".join(descs))


# --- Sleep Architecture ---
with tab_arch:
    col_a, col_b = st.columns([1, 1.2])
    with col_a:
        palette = get_plot_palette()
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        fig.patch.set_facecolor(palette['fig_bg'])
        ax.set_facecolor(palette['fig_bg'])
        plot_architecture_pie(ax, report['Counts'], palette=palette)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_b:
        st.markdown("##### Stage durations")
        rows = []
        for i, name in enumerate(STAGES):
            n_ep = report['Counts'][name]
            mins = n_ep * EPOCH_DURATION / 60.0
            pct = mins / max(1, report['TST_min'] if i > 0 else report['TIB_min']) * 100
            rows.append({
                "Stage": STAGE_FULL[i],
                "Epochs": n_ep,
                "Duration": fmt_hms(mins),
                "% of " + ("TST" if i > 0 else "TIB"): f"{pct:.1f} %",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.markdown("##### Sleep continuity")
        st.markdown(f"""
        - **Sleep cycles detected**: {report['Cycles']}
        - **Stage transitions**: {report['Transitions']}
        - **Fragmentation index**: {report['Fragmentation_idx']:.1f} (transitions per 100 sleep epochs)
        - **Mean prediction confidence**: {confs.mean()*100:.1f} %
        """)


# --- AI vs Expert ---
with tab_perf:
    if y_true is None:
        st.info("Upload an expert hypnogram (`.edf`) to enable validation metrics.")
    elif np.all(y_true == -1):
        st.warning("Expert hypnogram could not be parsed — see Hypnogram tab for details.")
    else:
        c_a, c_b = st.columns([1, 1])
        with c_a:
            palette = get_plot_palette()
            fig, ax = plt.subplots(figsize=(5.5, 5))
            fig.patch.set_facecolor(palette['fig_bg'])
            ax.set_facecolor(palette['fig_bg'])
            plot_confusion(ax, y_true, preds, palette=palette)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
        with c_b:
            st.markdown("##### Per-stage performance")
            df_m = per_class_metrics(y_true, preds)
            st.dataframe(df_m, hide_index=True, use_container_width=True)
            st.markdown("##### Global agreement")
            valid = y_true != -1
            ko = cohens_kappa(y_true, preds)
            kappa_label = (
                "Almost perfect" if ko >= 0.81 else
                "Substantial" if ko >= 0.61 else
                "Moderate" if ko >= 0.41 else
                "Fair" if ko >= 0.21 else "Poor"
            )
            st.metric("Overall accuracy", f"{overall_acc:.2f} %")
            st.metric("Cohen's κ", f"{ko:.3f}", delta=kappa_label, delta_color="off")
            st.metric("Epochs evaluated", f"{int(valid.sum())} / {n_epochs}")


# --- Clinical Report ---
with tab_report:
    st.markdown("##### Automated interpretation")
    interp = []
    se = report['SE_pct']
    if se >= 85: interp.append(("Sleep efficiency is **normal** (≥85%).", "norm"))
    elif se >= 75: interp.append((f"Sleep efficiency is **borderline** ({se:.1f}%).", "warn"))
    else: interp.append((f"Sleep efficiency is **reduced** ({se:.1f}%) — possible insomnia or fragmented sleep.", "alert"))

    if report['Latency_min'] < 5:
        interp.append((f"Very short sleep onset latency ({report['Latency_min']:.1f} min) — suggests excessive sleepiness.", "warn"))
    elif report['Latency_min'] > 30:
        interp.append((f"Prolonged sleep onset latency ({report['Latency_min']:.1f} min) — possible insomnia.", "warn"))

    if report['REM_pct'] < 15:
        interp.append((f"REM sleep is **reduced** ({report['REM_pct']:.1f}% of TST). Consider REM suppression.", "warn"))
    elif report['REM_pct'] > 30:
        interp.append((f"REM sleep is **elevated** ({report['REM_pct']:.1f}% of TST).", "warn"))
    else:
        interp.append((f"REM sleep proportion is within normal range ({report['REM_pct']:.1f}%).", "norm"))

    if report['N3_pct'] < 10:
        interp.append((f"Slow-wave sleep (N3) is **markedly reduced** ({report['N3_pct']:.1f}%).", "alert"))
    elif report['N3_pct'] < 13:
        interp.append((f"Slow-wave sleep (N3) is **low** ({report['N3_pct']:.1f}%).", "warn"))
    else:
        interp.append((f"Slow-wave sleep (N3) is within expected range ({report['N3_pct']:.1f}%).", "norm"))

    if report['Awakenings'] > 15:
        interp.append((f"**High number of awakenings** ({report['Awakenings']}) — indicates fragmented sleep.", "alert"))

    if report['Cycles'] < 3:
        interp.append((f"Only {report['Cycles']} sleep cycle(s) detected — typically 4–6 in adults.", "warn"))

    for txt, sev in interp:
        chip = {"norm": "chip-norm", "warn": "chip-warn", "alert": "chip-alert"}[sev]
        st.markdown(
            f"<div class='interp-box {sev}'>"
            f"<span class='chip {chip}'>{sev.upper()}</span> &nbsp; {txt}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("##### Quantitative summary")
    summary_df = pd.DataFrame([
        {"Metric": "Total Sleep Time (TST)", "Value": fmt_hms(report['TST_min']),    "Reference": "6.5 – 8h"},
        {"Metric": "Sleep Efficiency",       "Value": f"{report['SE_pct']:.1f} %",   "Reference": "≥ 85 %"},
        {"Metric": "Sleep Onset Latency",    "Value": fmt_hms(report['Latency_min']),"Reference": "< 20 min"},
        {"Metric": "REM Latency",            "Value": fmt_hms(report['REM_lat_min']),"Reference": "70 – 120 min"},
        {"Metric": "WASO",                   "Value": fmt_hms(report['WASO_min']),   "Reference": "< 30 min"},
        {"Metric": "N1 % of TST",            "Value": f"{report['N1_pct']:.1f} %",   "Reference": "2 – 5 %"},
        {"Metric": "N2 % of TST",            "Value": f"{report['N2_pct']:.1f} %",   "Reference": "45 – 55 %"},
        {"Metric": "N3 % of TST",            "Value": f"{report['N3_pct']:.1f} %",   "Reference": "13 – 23 %"},
        {"Metric": "REM % of TST",           "Value": f"{report['REM_pct']:.1f} %",  "Reference": "20 – 25 %"},
        {"Metric": "Number of awakenings",   "Value": str(report['Awakenings']),     "Reference": "< 10"},
        {"Metric": "Sleep cycles",           "Value": str(report['Cycles']),         "Reference": "4 – 6"},
    ])
    st.dataframe(summary_df, hide_index=True, use_container_width=True)

    st.markdown("##### Export")
    csv_bytes = make_report_csv(preds, y_true)
    st.download_button(
        "⬇ Download epoch-by-epoch CSV", data=csv_bytes,
        file_name=f"{patient_id}_hypnogram.csv", mime="text/csv",
    )


# --- Technical ---
with tab_tech:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Signal acquisition")
        st.json({
            "File": os.path.basename(sig_path),
            "Anonymous ID": patient_id,
            "Recording date": rec_date,
            "Channels in EDF": n_channels,
            "EOG channel selected": eog_ch,
            "Original sampling rate": f"{sfreq_orig:.0f} Hz",
            "Resampled to": f"{SFREQ} Hz",
            "Bandpass filter": "FIR 0.5 – 35 Hz",
            "Normalization": "Z-score + clip ±3σ",
            "Epoch length": f"{int(EPOCH_DURATION)} s ({N_POINTS} samples)",
            "Total epochs": n_epochs,
            "Recording duration": fmt_hms(duration_min),
        })
    with c2:
        st.markdown("##### Inference engine")
        st.json({
            "Model": "1D-CNN (NPU-compatible)",
            "Format": "OpenVINO IR FP16",
            "Parameters": "455 557",
            "Static input shape": f"({BATCH_INFER}, {N_POINTS}, 1)",
            "Output classes": 5,
            "Selected device": device_choice,
            "Runtime device": exec_dev,
            "Hardware": hw_name,
            "Inference time": f"{inf_t*1000:.1f} ms",
            "Throughput": f"{n_epochs/inf_t:.0f} epochs/s",
            "Mean confidence": f"{confs.mean()*100:.1f} %",
        })

    st.markdown("---")
    st.markdown("""
    <div class='disclaimer'>
    <b>Limitations.</b> The model uses a single EOG channel only and was trained on adult recordings.
    Performance may differ on pediatric subjects or in the presence of severe pathologies (REM behavior disorder, narcolepsy).
    Clinical decisions must rely on expert visual scoring of the full polysomnography montage.
    </div>
    """, unsafe_allow_html=True)

theme_style_last()
