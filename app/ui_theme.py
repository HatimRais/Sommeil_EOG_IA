"""
Single source of truth for DeepSleep AI dashboard colors (Streamlit + Matplotlib).
Clinical sleep-stage hues are fixed AASM-style; UI tokens follow a semantic light/dark scale.
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

# ----- Clinical (AASM-consistent, identical in all themes) -----
STAGE_COLOR_HEX: Dict[int, str] = {
    0: "#EAB308",  # W  — warm yellow (AASM yellow band)
    1: "#93C5E8",  # N1 — light blue
    2: "#3B82B6",  # N2 — medium blue
    3: "#0C4A6E",  # N3 — deep blue
    4: "#B91C1C",  # REM — red
}
STAGE_NAMES_SHORT = ("W", "N1", "N2", "N3", "REM")

# UI theme mode (sidebar radio)
ThemeMode = Literal["System", "Light", "Dark"]


def _hex_to_rgb(s: str) -> Tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def text_on_background(bg_hex: str) -> str:
    """WCAG-style contrast: dark text on light bg, light text on dark bg."""
    r, g, b = _hex_to_rgb(bg_hex)
    # relative luminance (sRGB)
    def lin(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    return "#0F172A" if L > 0.45 else "#F8FAFC"


# ---------------------------------------------------------------------------
# CSS custom properties (light / dark) — all dashboard colors derive from here
# ---------------------------------------------------------------------------
def _var_block_light() -> str:
    return """
    --dps-bg:            #EEF1F5;
    --dps-surface:       #FFFFFF;
    --dps-surface-2:     #F4F6F9;
    --dps-border:        #D2DAE3;
    --dps-border-soft:   #E4E9EF;
    --dps-text:          #0F172A;
    --dps-text-muted:    #5C6B7A;
    --dps-primary:       #0C4A6E;
    --dps-primary-mid:   #1E5F8A;
    --dps-primary-soft:  #E0EDF4;
    --dps-accent:        #1E5F8A;
    --dps-success:       #0F766E;
    --dps-warning:       #A16207;
    --dps-danger:        #B91C1C;
    --dps-on-primary:    #FFFFFF;
    --dps-shadow:        0 1px 2px rgba(15, 23, 42, 0.05), 0 2px 6px rgba(15, 23, 42, 0.06);
    --dps-shadow-lg:     0 4px 14px rgba(15, 23, 42, 0.08);
    --dps-sb-bg:         #E4E8EE;
    --dps-sb-surface:    #FFFFFF;
    --dps-sb-border:     #C8D0DA;
    --dps-sb-text:       #0F172A;
    --dps-sb-muted:      #5C6B7A;
    --dps-sb-accent:     #0C4A6E;
    --dps-header-from:   #0C4A6E;
    --dps-header-to:     #1E5F8A;
    --dps-header-text:   #FFFFFF;
    --dps-chip-norm-bg:  #ECFDF5;  --dps-chip-norm-fg:  #047857;  --dps-chip-norm-bd:  #6EE7B7;
    --dps-chip-warn-bg:  #FFFBEB;  --dps-chip-warn-fg:  #A16207;  --dps-chip-warn-bd:  #FCD34D;
    --dps-chip-alert-bg: #FEF2F2;  --dps-chip-alert-fg: #B91C1C;  --dps-chip-alert-bd: #FCA5A5;
    --dps-chip-info-bg:  #EFF6FF;  --dps-chip-info-fg:  #1D4ED8;  --dps-chip-info-bd:  #93C5FD;
    """


def _var_block_dark() -> str:
    return """
    --dps-bg:            #0B1020;
    --dps-surface:       #121A2C;
    --dps-surface-2:     #182236;
    --dps-border:        #2A3447;
    --dps-border-soft:   #1E2838;
    --dps-text:          #E8EDF3;
    --dps-text-muted:    #94A3B8;
    --dps-primary:       #5BA3D0;
    --dps-primary-mid:   #6BB8D9;
    --dps-primary-soft:  #1A3048;
    --dps-accent:        #5EB8A8;
    --dps-success:       #2DD4BF;
    --dps-warning:       #FBBF24;
    --dps-danger:        #F87171;
    --dps-on-primary:    #0B1020;
    --dps-shadow:        0 1px 2px rgba(0,0,0,0.4), 0 2px 6px rgba(0,0,0,0.28);
    --dps-shadow-lg:     0 4px 16px rgba(0,0,0,0.45);
    --dps-sb-bg:         #070B14;
    --dps-sb-surface:    #121A2C;
    --dps-sb-border:     #2A3447;
    --dps-sb-text:       #E8EDF3;
    --dps-sb-muted:      #94A3B8;
    --dps-sb-accent:     #5EB8A8;
    --dps-header-from:   #0C4A6E;
    --dps-header-to:     #1E5F8A;
    --dps-header-text:   #F8FAFC;
    --dps-chip-norm-bg:  #0B2215;  --dps-chip-norm-fg:  #6EE7B7;  --dps-chip-norm-bd:  #14532D;
    --dps-chip-warn-bg:  #1E1808;  --dps-chip-warn-fg:  #FCD34D;  --dps-chip-warn-bd:  #854D0E;
    --dps-chip-alert-bg: #2A1212;  --dps-chip-alert-fg: #FCA5A5;  --dps-chip-alert-bd: #7F1D1D;
    --dps-chip-info-bg:  #0C1826;  --dps-chip-info-fg:  #93C5FD;  --dps-chip-info-bd:  #1E3A8A;
    """


def css_root_variables(mode: ThemeMode) -> str:
    """Returns :root rules for the selected appearance mode."""
    light = _var_block_light().strip()
    dark = _var_block_dark().strip()
    if mode == "Light":
        return f":root {{ {light} color-scheme: light; }}".strip()
    if mode == "Dark":
        return f":root {{ {dark} color-scheme: dark; }}".strip()
    return (
        f":root {{ {light} color-scheme: light; }}\n"
        f"@media (prefers-color-scheme: dark) {{ :root {{ {dark} color-scheme: dark; }} }}"
    ).strip()


# Legacy aliases in CSS: map --* used in templates to --dps-*
def css_legacy_bridge() -> str:
    return """
    :root {
      --bg: var(--dps-bg);
      --surface: var(--dps-surface);
      --surface-2: var(--dps-surface-2);
      --border: var(--dps-border);
      --border-soft: var(--dps-border-soft);
      --text: var(--dps-text);
      --text-muted: var(--dps-text-muted);
      --primary: var(--dps-primary);
      --primary-2: var(--dps-primary-mid);
      --primary-soft: var(--dps-primary-soft);
      --accent: var(--dps-accent);
      --success: var(--dps-success);
      --warning: var(--dps-warning);
      --danger: var(--dps-danger);
      --shadow: var(--dps-shadow);
      --shadow-hover: var(--dps-shadow-lg);
      --sb-bg: var(--dps-sb-bg);
      --sb-surface: var(--dps-sb-surface);
      --sb-border: var(--dps-sb-border);
      --sb-text: var(--dps-sb-text);
      --sb-muted: var(--dps-sb-muted);
      --sb-accent: var(--dps-sb-accent);
      --header-from: var(--dps-header-from);
      --header-to: var(--dps-header-to);
      --header-text: var(--dps-header-text);
      --chip-norm-bg: var(--dps-chip-norm-bg);
      --chip-norm-fg: var(--dps-chip-norm-fg);
      --chip-norm-bd: var(--dps-chip-norm-bd);
      --chip-warn-bg: var(--dps-chip-warn-bg);
      --chip-warn-fg: var(--dps-chip-warn-fg);
      --chip-warn-bd: var(--dps-chip-warn-bd);
      --chip-alert-bg: var(--dps-chip-alert-bg);
      --chip-alert-fg: var(--dps-chip-alert-fg);
      --chip-alert-bd: var(--dps-chip-alert-bd);
      --chip-info-bg: var(--dps-chip-info-bg);
      --chip-info-fg: var(--dps-chip-info-fg);
      --chip-info-bd: var(--dps-chip-info-bd);
    }
    """


DASHBOARD_BASE_CSS = """
html, body, #root, [class*="css"] { font-family: 'Segoe UI', 'Inter', system-ui, sans-serif; }
/* Évite une bande noire (fond natif) au-dessus de la zone principale / .med-header */
html, body { background: var(--dps-bg) !important; background-color: var(--dps-bg) !important; }
.stApp { background: var(--dps-bg) !important; color: var(--dps-text) !important; }
[data-testid="stAppViewContainer"] {
  background: var(--dps-bg) !important; background-color: var(--dps-bg) !important;
  color: var(--dps-text) !important; padding-top: 0 !important;
}
/* Enveloppe du contenu (évite le ruban sombre hérité du thème Streamlit) */
[data-testid="stAppViewBlockContainer"] {
  background: var(--dps-bg) !important; background-color: var(--dps-bg) !important; padding-top: 0 !important;
}
/* Bandeau hamburger / outils (souvent `header` en haut de la vue) */
.stApp [data-testid="stAppViewContainer"] > header,
.stApp [data-testid="stAppViewBlockContainer"] > [data-testid="stHeader"],
section.main [data-testid="stHeader"] {
  background: var(--dps-bg) !important; background-color: var(--dps-bg) !important;
  color: var(--dps-text) !important; border: none !important; box-shadow: none !important;
}
section.main, [data-testid="stMain"] {
  background: transparent !important; padding-top: 0 !important; margin-top: 0 !important;
}
/* Contenu : premier bloc aligné, sans chevauchement visuel */
.block-container { padding-top: 0.5rem; padding-bottom: 2rem; max-width: 1500px; }
.med-header { margin-top: 0.25rem; position: relative; z-index: 1; }
h1, h2, h3, h4, h5, h6, p, li, label { color: var(--dps-text) !important; -webkit-text-fill-color: var(--dps-text) !important; }
a { color: var(--dps-primary-mid) !important; }
code { background: var(--dps-surface-2) !important; color: var(--dps-text) !important; border: 1px solid var(--dps-border-soft); border-radius: 4px; padding: 1px 5px; }

/* Sidebar (no universal * — it forced dark text on Streamlit’s dark stHeader/ezrtsby2 blocks) */
[data-testid="stSidebar"] { background: var(--dps-sb-bg) !important; border-right: 1px solid var(--dps-sb-border); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--dps-sb-text) !important; -webkit-text-fill-color: var(--dps-sb-text) !important; }
[data-testid="stSidebar"] h5, [data-testid="stSidebar"] h3 { color: var(--dps-sb-accent) !important; -webkit-text-fill-color: var(--dps-sb-accent) !important; }
/* Section / widget group headers: Streamlit wraps labels in stHeader + dark Base Web divs (ezrtsby2) */
[data-testid="stSidebar"] [data-testid="stHeader"] {
  background: transparent !important; background-color: transparent !important;
  padding: 0 !important; margin: 0 !important; border: none !important; box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stHeader"] h1, [data-testid="stSidebar"] [data-testid="stHeader"] h2, [data-testid="stSidebar"] [data-testid="stHeader"] h3 {
  color: var(--dps-sb-accent) !important; -webkit-text-fill-color: var(--dps-sb-accent) !important; font-size: 0.72rem !important;
  text-transform: uppercase; letter-spacing: 1.1px; font-weight: 700 !important; margin: 0 !important; padding: 0 !important;
}
[data-testid="stSidebar"] [class*="ezrtsby2"] {
  background: var(--dps-sb-surface) !important; background-color: var(--dps-sb-surface) !important;
  color: var(--dps-sb-text) !important; -webkit-text-fill-color: var(--dps-sb-text) !important;
  box-shadow: none !important; border: none !important; border-radius: 8px;
}
[data-testid="stSidebar"] [class*="ezrtsby2"] p, [data-testid="stSidebar"] [class*="ezrtsby2"] span,
[data-testid="stSidebar"] [class*="ezrtsby2"] [data-testid="stMarkdownContainer"] p {
  color: var(--dps-sb-text) !important; -webkit-text-fill-color: var(--dps-sb-text) !important;
}
/* Widget label text (visible on all themes) */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label { color: var(--dps-sb-text) !important; -webkit-text-fill-color: var(--dps-sb-text) !important; }
/* Brand block in sidebar (custom HTML) */
.dps-sidebar-brand { margin: 0.15rem 0 1.1rem 0; }
.dps-sidebar-brand .dps-mark { display: flex; align-items: center; gap: 0.75rem; }
.dps-sidebar-brand .dps-ico { font-size: 2.65rem; line-height: 1; }
.dps-sidebar-brand .dps-titles h2 { color: var(--dps-sb-text) !important; -webkit-text-fill-color: var(--dps-sb-text) !important; margin: 0; font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; }
.dps-sidebar-brand .dps-titles p { color: var(--dps-sb-muted) !important; -webkit-text-fill-color: var(--dps-sb-muted) !important; margin: 0.2rem 0 0; font-size: 0.78rem; }
.dps-sidebar-brand .dps-badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.6rem; }
.dps-sidebar-brand .dps-badge { font-size: 0.64rem; font-weight: 600; padding: 2px 8px; border-radius: 999px; background: var(--dps-primary-soft) !important; color: var(--dps-sb-accent) !important; border: 1px solid var(--dps-sb-border) !important; }
.dps-sidebar-hint { font-size: 0.75rem; color: var(--dps-sb-muted) !important; margin: 0.85rem 0 0.5rem; line-height: 1.4; }
.dps-sidebar-hint strong { color: var(--dps-sb-text) !important; -webkit-text-fill-color: var(--dps-sb-text) !important; font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stExpander"] { background: var(--dps-sb-surface) !important; border: 1px solid var(--dps-sb-border) !important; border-radius: 8px; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary, [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li { color: var(--dps-sb-text) !important; -webkit-text-fill-color: var(--dps-sb-text) !important; }
[data-testid="stSidebar"] hr { border: 0; border-top: 1px solid var(--dps-sb-border); margin: 0.6rem 0; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div { background: var(--dps-sb-surface) !important; border-color: var(--dps-sb-border) !important; color: var(--dps-sb-text) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] input { color: var(--dps-sb-text) !important; -webkit-text-fill-color: var(--dps-sb-text) !important; }
[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: var(--dps-sb-accent) !important; }
[data-baseweb="popover"] [role="listbox"] { background: var(--dps-sb-surface) !important; border: 1px solid var(--dps-sb-border) !important; }
[data-baseweb="popover"] [role="option"] { color: var(--dps-sb-text) !important; background: var(--dps-sb-surface) !important; }
[data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] [role="option"][aria-selected="true"] { background: var(--dps-primary-soft) !important; }
[data-testid="stSidebar"] .stRadio > div { background: var(--dps-sb-surface) !important; border: 1px solid var(--dps-sb-border) !important; border-radius: 8px; }
[data-testid="stSidebar"] .stRadio [role="radiogroup"]:not([style*="row"]) > label[data-checked="true"] { background: var(--dps-primary-soft) !important; border: 1px solid var(--dps-sb-accent) !important; }
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] { background: var(--dps-sb-surface) !important; border: 1px solid var(--dps-sb-border) !important; border-radius: 8px; padding: 3px; }
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] > label { color: var(--dps-sb-text) !important; }
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] > label[data-checked="true"] { background: var(--dps-sb-accent) !important; color: var(--dps-sb-bg) !important; -webkit-text-fill-color: var(--dps-sb-bg) !important; }
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"][style*="row"] > label > div:first-child { display: none !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section { background: var(--dps-sb-surface) !important; border: 1.5px dashed var(--dps-sb-border) !important; border-radius: 8px; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] button { background: var(--dps-sb-accent) !important; color: var(--dps-on-primary) !important; }
[data-testid="stSidebar"] .stButton > button { background: var(--dps-sb-accent) !important; color: var(--dps-on-primary) !important; font-weight: 600; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background: var(--dps-primary) !important; color: #FFFFFF !important; }

/* Header & cards */
.med-header {
  background: linear-gradient(90deg, var(--dps-header-from) 0%, var(--dps-header-to) 100%);
  color: var(--dps-header-text);
  padding: 1rem 1.4rem; border-radius: 10px; display: flex; justify-content: space-between; align-items: center;
  box-shadow: var(--dps-shadow); margin-bottom: 1.1rem;
}
.med-header h1 { color: var(--dps-header-text) !important; -webkit-text-fill-color: var(--dps-header-text) !important; margin: 0; font-size: 1.3rem; font-weight: 600; }
.med-header .subtitle { color: rgba(248, 250, 252, 0.8); font-size: 0.8rem; margin-top: 2px; }
.med-header .tag { background: rgba(255,255,255,0.14); color: var(--dps-header-text); padding: 4px 11px; border-radius: 999px; font-size: 0.75rem; border: 1px solid rgba(255,255,255,0.2); margin-left: 6px; }
.patient-card {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;
  background: var(--dps-surface); border: 1px solid var(--dps-border); border-left: 4px solid var(--dps-primary);
  border-radius: 8px; padding: 1rem 1.3rem; margin-bottom: 1.1rem; box-shadow: var(--dps-shadow);
}
.patient-card .label { font-size: 0.68rem; color: var(--dps-text-muted) !important; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
.patient-card .value { font-size: 0.95rem; color: var(--dps-text) !important; font-weight: 500; }
.section-title {
  font-size: 0.9rem; font-weight: 700; color: var(--dps-primary) !important; margin: 1.4rem 0 0.75rem; padding-bottom: 6px;
  border-bottom: 2px solid var(--dps-primary-mid); text-transform: uppercase; letter-spacing: 0.35px;
}

/* Metrics & chips & tabs */
div[data-testid="stMetric"] { background: var(--dps-surface) !important; border: 1px solid var(--dps-border) !important; border-radius: 8px; padding: 0.65rem 0.9rem; box-shadow: var(--dps-shadow); }
div[data-testid="stMetricLabel"] { color: var(--dps-text-muted) !important; -webkit-text-fill-color: var(--dps-text-muted) !important; }
div[data-testid="stMetricValue"] { color: var(--dps-primary) !important; -webkit-text-fill-color: var(--dps-primary) !important; }
.chip { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.35px; border: 1px solid; }
.chip-norm  { background: var(--dps-chip-norm-bg);  color: var(--dps-chip-norm-fg);  border-color: var(--dps-chip-norm-bd); }
.chip-warn  { background: var(--dps-chip-warn-bg);  color: var(--dps-chip-warn-fg);  border-color: var(--dps-chip-warn-bd); }
.chip-alert { background: var(--dps-chip-alert-bg); color: var(--dps-chip-alert-fg); border-color: var(--dps-chip-alert-bd); }
.chip-info  { background: var(--dps-chip-info-bg);  color: var(--dps-chip-info-fg);  border-color: var(--dps-chip-info-bd); }
.muted { color: var(--dps-text-muted) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--dps-border); }
.stTabs [data-baseweb="tab"] { color: var(--dps-text-muted) !important; font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--dps-primary) !important; border-bottom: 2px solid var(--dps-primary) !important; font-weight: 600; }
.stButton > button, .stDownloadButton > button { background: var(--dps-primary) !important; color: #FFFFFF !important; border: none; border-radius: 6px; }

/* Data & misc */
[data-testid="stDataFrame"] { border: 1px solid var(--dps-border); border-radius: 6px; background: var(--dps-surface) !important; }
[data-testid="stDataFrame"] table, [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td { background: var(--dps-surface) !important; color: var(--dps-text) !important; border-color: var(--dps-border) !important; }
[data-testid="stDataFrame"] thead th { background: var(--dps-surface-2) !important; }
.stProgress > div > div > div > div { background: var(--dps-primary-mid) !important; }
.disclaimer { background: var(--dps-chip-warn-bg); border-left: 4px solid var(--dps-warning); color: var(--dps-chip-warn-fg) !important; border-radius: 4px; padding: 9px 12px; font-size: 0.8rem; margin-top: 1.2rem; }
.interp-box { background: var(--dps-surface-2); border-left: 3px solid var(--dps-primary); color: var(--dps-text) !important; border-radius: 4px; padding: 9px 12px; margin-bottom: 5px; }
.interp-box.norm  { border-left-color: var(--dps-success); }
.interp-box.warn  { border-left-color: var(--dps-warning); }
.interp-box.alert { border-left-color: var(--dps-danger); }
span.med-legend-txt { color: var(--dps-text) !important; -webkit-text-fill-color: var(--dps-text) !important; }

/* Alerts: align with our semantic colors */
div[data-testid="stAlert"] { background: var(--dps-surface) !important; border: 1px solid var(--dps-border) !important; }
div[data-testid="stAlert"] p, div[data-testid="stAlert"] span { color: var(--dps-text) !important; }
"""

DASHBOARD_RESPONSIVE_CSS = """
@media (max-width: 900px) { .block-container { max-width: 100% !important; padding-left: 1rem; padding-right: 1rem; } }
@media (max-width: 768px) {
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] [data-testid="column"] { min-width: 0 !important; }
}
"""

# Strong overrides (run once at end of script too — wins over Streamlit emotion)
DASHBOARD_TAIL_CSS = """
html, body { background: var(--dps-bg) !important; background-color: var(--dps-bg) !important; }
[data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] { background: var(--dps-bg) !important; padding-top: 0 !important; }
.stApp [data-testid="stAppViewContainer"] > header,
[data-testid="stToolbar"] {
  background: var(--dps-bg) !important; background-color: var(--dps-bg) !important;
  color: var(--dps-text) !important;
}
.med-header { z-index: 2; }
section.main [data-testid="stMarkdownContainer"] p,
section.main [data-testid="stMarkdownContainer"] li,
section.main [data-testid="stMarkdownContainer"] span,
[role="tabpanel"] [data-testid="stMarkdownContainer"] p,
[role="tabpanel"] [data-testid="stMarkdownContainer"] li { color: var(--dps-text) !important; -webkit-text-fill-color: var(--dps-text) !important; }
/* Reinforce: sidebar group headers (Streamlit emotion + Base Web) */
[data-testid="stSidebar"] [class*="ezrtsby2"] { background: var(--dps-sb-surface) !important; color: var(--dps-sb-text) !important; }
[data-testid="stSidebar"] [data-testid="stHeader"] h1, [data-testid="stSidebar"] [data-testid="stHeader"] h2, [data-testid="stSidebar"] [data-testid="stHeader"] h3 { color: var(--dps-sb-accent) !important; -webkit-text-fill-color: var(--dps-sb-accent) !important; }
"""


def build_dashboard_stylesheet(mode: ThemeMode) -> str:
    parts = [
        css_root_variables(mode),
        css_legacy_bridge(),
        DASHBOARD_BASE_CSS,
        DASHBOARD_RESPONSIVE_CSS,
        DASHBOARD_TAIL_CSS,
    ]
    return "\n".join(parts)


# ----- Matplotlib palettes (tied to light/dark figure background) -----
def resolve_plot_uses_light_charts(session_theme: ThemeMode) -> bool:
    """
    Server-side: System OS preference is unknown in Streamlit <1.40 without JS.
    Light / System → light charts; Dark → dark charts.
    """
    return session_theme in ("System", "Light")


def get_matplotlib_plot_dict(light: bool) -> Dict[str, Any]:
    """Single dict for hypnogram, pie, confusion — clinical stage colors always STAGE_COLOR_HEX."""
    stg = {i: STAGE_COLOR_HEX[i] for i in range(5)}
    if light:
        return dict(
            fig_bg="#FFFFFF",
            axes_bg="#F8FAFC",
            text="#0F172A",
            muted="#5C6B7A",
            spine="#C8D1DC",
            grid="#E2E8F0",
            primary="#0C4A6E",
            primary_2="#3B82B6",
            pie_edge="#FFFFFF",
            cm_text_dark="#0F172A",
            cm_text_light="#F8FAFC",
            stage_colors=stg,
            cmap="Blues",
        )
    return dict(
        fig_bg="#121A2C",
        axes_bg="#0B1020",
        text="#E8EDF3",
        muted="#94A3B8",
        spine="#3A4A62",
        grid="#1E2A3D",
        primary="#5BA3D0",
        primary_2="#5EB8A8",
        pie_edge="#121A2C",
        cm_text_dark="#0B1020",
        cm_text_light="#F8FAFC",
        stage_colors=stg,
        cmap="Blues",
    )
