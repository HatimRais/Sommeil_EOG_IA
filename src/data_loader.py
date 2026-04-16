import mne
import os


def load_eog_signal(file_path):
    """
    Charge un fichier EDF et extrait le canal EOG.
    """
    # Lecture du fichier EDF
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Sélection du canal EOG (souvent nommé 'EOG horizontal' ou 'eog')
    # On filtre pour ne garder que ce qui nous intéresse
    eog_channels = [ch for ch in raw.ch_names if 'EOG' in ch.upper()]
    raw.pick_channels(eog_channels)

    print(f"Canaux extraits : {eog_channels}")
    return raw


if __name__ == "__main__":
    # Testez avec l'un de vos fichiers dans data/raw/
    path = "data/raw/votre_fichier_test.edf"
    if os.path.exists(path):
        signal = load_eog_signal(path)
        # Visualisation rapide des 10 premières secondes
        signal.plot(duration=10, n_channels=1, scalings='auto')