import re
import mne
import numpy as np

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


def create_dataset(psg_path: str, hypno_path: str):
    """
    Pipeline d'entraînement : synchronise signal EOG + hypnogramme.
    Retourne (X, y) avec X de forme (n_epochs, 1, 3000) et y (n_epochs,).
    """
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    raw.pick_channels(['EOG horizontal'])
    raw.filter(l_freq=0.3, h_freq=35.0, verbose=False)

    annotations = mne.read_annotations(hypno_path)
    raw.set_annotations(annotations, emit_warning=False)

    annotation_desc_2_event_id = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4,
    }

    events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, chunk_duration=30.0, verbose=False
    )

    epochs = mne.Epochs(
        raw, events, event_id=annotation_desc_2_event_id,
        tmin=0., tmax=30. - 1. / raw.info['sfreq'],
        baseline=None, preload=True, verbose=False,
    )

    X = epochs.get_data()
    y = epochs.events[:, 2]
    return X, y


if __name__ == "__main__":
    psg = "../data/raw/Patient_01_Signal.edf"
    hyp = "../data/raw/Patient_01_Labels.edf"
    X, y = create_dataset(psg, hyp)
    print(f"Époques : {len(X)} | Forme X : {X.shape} | Distribution : {np.bincount(y)}")
