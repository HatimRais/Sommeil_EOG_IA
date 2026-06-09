import os
import re
import glob
from typing import List, Optional, Tuple

import mne
import numpy as np

# Fréquence cible (alignée modèle NPU / dashboard)
TARGET_SFREQ = 100.0
EPOCH_DURATION = 30.0
N_SAMPLES_PER_EPOCH = int(TARGET_SFREQ * EPOCH_DURATION)

# Libellés MNE standard (Sleep-EDF Expanded, PhysioNet)
ANNOTATION_EVENT_ID = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

# Mapping AASM (5 classes) — N3 et N4 fusionnés
STAGE_MAP = {
    'W': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 3,
    'R': 4,
}

UNKNOWN_TOKENS = {'?', 'MOVEMENT', 'UNKNOWN', 'NONE'}

# Regex tolérante : "Sleep stage W", "Stage 1", "SLEEP-STAGE-R", "stage_2", "W"
_STAGE_RE = re.compile(r'(?:SLEEP[\s_-]*)?STAGE[\s_-]*([W1234R?])', re.IGNORECASE)


def parse_stage(description: str) -> int:
    """
    Convertit une description d'annotation EDF en code de stade.
    Retourne le code (0-4) ou -1 si non reconnu / inconnu.
    """
    if description is None:
        return -1

    desc = str(description).strip().upper()

    if any(tok in desc for tok in UNKNOWN_TOKENS):
        if '?' in desc or 'UNKNOWN' in desc:
            return -1

    m = _STAGE_RE.search(desc)
    if m:
        token = m.group(1).upper()
        if token == '?':
            return -1
        return STAGE_MAP.get(token, -1)

    if desc in STAGE_MAP:
        return STAGE_MAP[desc]

    return -1


def load_and_sync_labels(label_path: str, n_epochs: int, epoch_duration: float = 30.0):
    """
    Charge un EDF d'annotations et produit un vecteur de labels par époque.

    Returns
    -------
    y_true : np.ndarray of shape (n_epochs,)
        Codes de stade (0..4) ou -1 pour les époques sans label valide.
        Retourne None si le fichier ne peut pas être lu du tout.
    """
    try:
        annots = mne.read_annotations(label_path)
    except Exception as exc:
        print(f"[load_and_sync_labels] Erreur de lecture EDF : {exc}")
        return None

    y_true = np.full(n_epochs, -1, dtype=int)

    if len(annots) == 0:
        return y_true

    for a in annots:
        stage = parse_stage(a['description'])
        if stage < 0:
            continue

        start_idx = int(round(a['onset'] / epoch_duration))
        n_steps = max(1, int(round(a['duration'] / epoch_duration)))
        end_idx = min(start_idx + n_steps, n_epochs)

        if start_idx < n_epochs and end_idx > start_idx:
            y_true[start_idx:end_idx] = stage

    return y_true


def annotation_summary(label_path: str):
    """
    Outil de debug : retourne (descriptions_uniques, n_total) du fichier d'annotations.
    """
    try:
        annots = mne.read_annotations(label_path)
        return sorted(set(annots.description)), len(annots)
    except Exception as exc:
        return [f"<erreur: {exc}>"], 0


def _subject_key_from_filename(filename: str) -> Optional[str]:
    """Ex. SC4001E0-PSG.edf → SC4001 ; ST7012J0-PSG.edf → ST7012."""
    m = re.match(r"(SC\d{4}|ST\d{4})", os.path.basename(filename))
    return m.group(1) if m else None


def pick_eog_channel(raw: mne.io.BaseRaw) -> str:
    """Sélectionne un canal EOG (Sleep-EDF : « EOG horizontal », autres : nom contenant EOG)."""
    for name in raw.ch_names:
        if name.strip().lower() == "eog horizontal":
            return name
    for name in raw.ch_names:
        if "EOG" in name.upper():
            return name
    raise ValueError(f"Aucun canal EOG dans {raw.ch_names}")


def discover_sleep_edf_pairs(root_dir: str) -> List[Tuple[str, str]]:
    """
    Associe chaque *-PSG.edf à son *-Hypnogram.edf (même dossier, même id sujet 6 car.).
    Compatible Sleep-EDF Expanded (cassette + telemetry).
    """
    pairs: List[Tuple[str, str]] = []
    for psg in sorted(glob.glob(os.path.join(root_dir, "**", "*-PSG.edf"), recursive=True)):
        folder = os.path.dirname(psg)
        sid = _subject_key_from_filename(psg)
        if not sid:
            continue
        hyps = [
            h
            for h in glob.glob(os.path.join(folder, "*-Hypnogram.edf"))
            if _subject_key_from_filename(h) == sid
        ]
        if len(hyps) == 1:
            pairs.append((psg, hyps[0]))
    return pairs


def discover_legacy_raw_pairs(raw_dir: str) -> List[Tuple[str, str]]:
    """Paires Patient_XX_Signal.edf / Patient_XX_Labels.edf."""
    sigs = sorted(glob.glob(os.path.join(raw_dir, "*Signal.edf")))
    labs = sorted(glob.glob(os.path.join(raw_dir, "*Labels.edf")))
    return list(zip(sigs, labs))


def create_dataset(psg_path: str, hypno_path: str, resample: bool = True):
    """
    Pipeline d'entraînement : synchronise signal EOG + hypnogramme.
    Retourne (X, y) avec X de forme (n_epochs, 1, n_samples) et y (n_epochs,).
    """
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    eog_ch = pick_eog_channel(raw)
    raw.pick([eog_ch])
    if resample and raw.info["sfreq"] != TARGET_SFREQ:
        raw.resample(TARGET_SFREQ, verbose=False)
    raw.filter(l_freq=0.3, h_freq=35.0, verbose=False)

    annotations = mne.read_annotations(hypno_path)
    raw.set_annotations(annotations, emit_warning=False)

    events, _ = mne.events_from_annotations(
        raw,
        event_id=ANNOTATION_EVENT_ID,
        chunk_duration=EPOCH_DURATION,
        verbose=False,
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=ANNOTATION_EVENT_ID,
        tmin=0.0,
        tmax=EPOCH_DURATION - 1.0 / raw.info["sfreq"],
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data(copy=True)
    y = epochs.events[:, 2]
    return X, y


def load_corpus(
    sleep_edf_root: Optional[str] = None,
    legacy_raw_dir: Optional[str] = None,
    normalize_per_subject: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Charge tous les sujets disponibles.

    Returns
    -------
    X : (n_epochs, n_samples, 1) float32
    y : (n_epochs,) int
    subject_idx : (n_epochs,) int — indice du sujet pour split par patient
    subject_names : liste des identifiants sujet
    """
    pairs: List[Tuple[str, str]] = []
    subject_names: List[str] = []

    if sleep_edf_root and os.path.isdir(sleep_edf_root):
        for psg, hyp in discover_sleep_edf_pairs(sleep_edf_root):
            pairs.append((psg, hyp))
            subject_names.append(os.path.basename(psg).replace("-PSG.edf", ""))

    if legacy_raw_dir and os.path.isdir(legacy_raw_dir):
        for psg, hyp in discover_legacy_raw_pairs(legacy_raw_dir):
            pairs.append((psg, hyp))
            subject_names.append(os.path.basename(psg).replace("_Signal.edf", ""))

    if not pairs:
        raise FileNotFoundError("Aucune paire PSG / Hypnogramme trouvée.")

    all_X, all_y, all_subj = [], [], []
    for i, (psg, hyp) in enumerate(pairs):
        if verbose:
            print(f"  [{i + 1}/{len(pairs)}] {subject_names[i]}", flush=True)
        try:
            X_sub, y_sub = create_dataset(psg, hyp)
            if normalize_per_subject:
                X_sub = (X_sub - np.mean(X_sub)) / (np.std(X_sub) + 1e-6)
            n = len(y_sub)
            all_X.append(X_sub)
            all_y.append(y_sub)
            all_subj.append(np.full(n, i, dtype=np.int32))
        except Exception as exc:
            if verbose:
                print(f"    SKIP : {exc}")

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int32)
    subject_idx = np.concatenate(all_subj, axis=0)

    # (epochs, 1, samples) → (epochs, samples, 1) pour Keras
    X = X.reshape(X.shape[0], X.shape[2], 1)

    if verbose:
        print(f"\nCorpus : {X.shape[0]} époques · {len(subject_names)} sujets")
        print(f"Distribution classes : {np.bincount(y, minlength=5)}")
    return X, y, subject_idx, subject_names


if __name__ == "__main__":
    psg = "../data/raw/Patient_01_Signal.edf"
    hyp = "../data/raw/Patient_01_Labels.edf"
    X, y = create_dataset(psg, hyp)
    print(f"Époques : {len(X)} | Forme X : {X.shape} | Distribution : {np.bincount(y)}")
