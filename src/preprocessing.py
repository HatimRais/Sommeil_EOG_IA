import numpy as np
import mne


def apply_preprocessing(raw):
    """
    Pipeline de nettoyage complet pour le signal EOG.
    """
    # 1. Resample à 100Hz (Standard pour le modèle)
    if raw.info['sfreq'] != 100.0:
        raw.resample(100.0)

    # 2. Sélection automatique du canal EOG
    eog_ch = [ch for ch in raw.ch_names if 'EOG' in ch.upper()][0]

    # 3. Filtrage FIR (0.5 - 35 Hz)
    raw.filter(0.5, 35.0, picks=[eog_ch], method='fir', verbose=False)
    data = raw.get_data(picks=[eog_ch])[0]

    # 4. Normalisation Z-score robuste et Clipping
    data = (data - np.mean(data)) / (np.std(data) + 1e-6)
    data = np.clip(data, -3, 3)

    return data, eog_ch