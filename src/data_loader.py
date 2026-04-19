import mne
import numpy as np


def create_dataset(psg_path, hypno_path):
    """
    Synchronise le signal EOG et l'hypnogramme pour créer les couples (X, y).
    """
    # 1. Chargement du signal EOG
    raw = mne.io.read_raw_edf(psg_path, preload=True)
    raw.pick_channels(['EOG horizontal'])  # Sélection du canal selon le dataset
    raw.filter(l_freq=0.3, h_freq=35.0)  # Filtrage selon le CDC

    # 2. Chargement des annotations (Labels)
    annotations = mne.read_annotations(hypno_path)
    raw.set_annotations(annotations, emit_warning=False)

    # 3. Mapping des stades du sommeil (Standard AASM) [cite: 39]
    # On regroupe les stades pour correspondre aux 5 classes : W, N1, N2, N3, REM
    annotation_desc_2_event_id = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,  # N3 et N4 sont souvent regroupés en Sommeil Profond
        'Sleep stage R': 4
    }

    # 4. Création des époques de 30 secondes [cite: 46]
    events, _ = mne.events_from_annotations(raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    # On découpe le signal en segments de 30s (tmin=0, tmax=30)
    # À 100Hz, cela donne 3000 points par segment
    epochs = mne.Epochs(raw, events, event_id=annotation_desc_2_event_id,
                        tmin=0., tmax=30. - 1. / raw.info['sfreq'],
                        baseline=None, preload=True)

    # 5. Extraction des données numériques (X) et des labels (y)
    X = epochs.get_data()  # Shape: (Nombre_d_époques, 1, 3000)
    y = epochs.events[:, 2]  # Shape: (Nombre_d_époques,)

    return X, y


if __name__ == "__main__":
    # Remplacez par vos chemins de fichiers réels dans data/raw/
    psg = "../data/raw/SC4001E0-PSG.edf"
    hyp = "../data/raw/SC4001EC-Hypnogram.edf"

    X, y = create_dataset(psg, hyp)

    print(f"Données chargées avec succès !")
    print(f"Nombre d'époques : {len(X)}")
    print(f"Forme de X (EOG) : {X.shape} (Epoques, Canaux, Points)")
    print(f"Distribution des classes (y) : {np.bincount(y)}")