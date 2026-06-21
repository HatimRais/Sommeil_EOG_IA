/** AASM sleep stage colors — fixed across themes (from ui_theme.py) */
export const STAGE_COLORS: Record<string, string> = {
  W: "#EAB308",
  N1: "#93C5E8",
  N2: "#3B82B6",
  N3: "#0C4A6E",
  REM: "#B91C1C",
};

export const STAGE_FULL: Record<string, string> = {
  W: "Éveil (Wake)",
  N1: "N1 (Stade 1)",
  N2: "N2 (Stade 2)",
  N3: "N3 (Sommeil profond)",
  REM: "REM",
};

export const NORMS = [
  { label: "Temps de sommeil total", range: "390–480 min" },
  { label: "Efficacité du sommeil", range: "≥ 85 %" },
  { label: "Latence d'endormissement", range: "< 20 min" },
  { label: "Éveils après endormissement", range: "< 30 min" },
  { label: "REM %", range: "20–25 %" },
  { label: "N3 (SWS) %", range: "13–23 %" },
  { label: "Latence REM", range: "70–120 min" },
];
