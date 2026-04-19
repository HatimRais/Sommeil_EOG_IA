import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from data_loader import create_dataset
from architecture import build_cnn_lstm_model

# 1. Configuration des chemins
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir = os.path.join(base_path, "data", "raw")
models_dir = os.path.join(base_path, "models")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# 2. Chargement automatique de TOUS les fichiers présents
print("--- Phase de chargement des données multi-sujets ---")
psg_files = sorted(glob.glob(os.path.join(raw_dir, "*PSG.edf")))
hyp_files = sorted(glob.glob(os.path.join(raw_dir, "*Hypnogram.edf")))

all_X = []
all_y = []

for p, h in zip(psg_files, hyp_files):
    print(f"Traitement du sujet : {os.path.basename(p)}")
    try:
        X_sub, y_sub = create_dataset(p, h)
        all_X.append(X_sub)
        all_y.append(y_sub)
    except Exception as e:
        print(f"Erreur sur le fichier {p}: {e}")

# Fusion de tous les tableaux
X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)

# Reshape pour le CNN : (Epoques, Points, Canal)
X = X.reshape(X.shape[0], X.shape[2], 1)

print(f"\nTotal des données chargées : {X.shape[0]} époques.")
print(f"Distribution des classes : {np.bincount(y)}")

# 3. Équilibrage des classes (Crucial pour le sommeil)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(weights))

# 4. Création et Entraînement
print("\n--- Initialisation de l'IA ---")
model = build_cnn_lstm_model(input_shape=(3000, 1))

print("\n--- Début de l'entraînement (Plusieurs sujets) ---")
# On passe à 20 époques car il y a plus de données à apprendre
history = model.fit(
    X, y,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    shuffle=True
)

# 5. Sauvegarde
model_save_path = os.path.join(models_dir, "sleep_model_v1.keras")
model.save(model_save_path)
print(f"\nModèle global sauvegardé : {model_save_path}")

# 6. Optimisation pour le NPU Intel AI Boost
print("\n--- Optimisation pour le NPU Intel ---")
try:
    import openvino as ov

    core = ov.Core()

    # Export pour conversion
    export_path = os.path.join(models_dir, "temp_export")
    model.export(export_path)

    ov_model = ov.convert_model(export_path)
    ov_xml_path = os.path.join(models_dir, "sleep_model_npu.xml")
    ov.save_model(ov_model, ov_xml_path)

    print(f"Succès ! Le NPU utilisera : {ov_xml_path}")
    print("Appareils Intel disponibles :", core.available_devices)
except Exception as e:
    print(f"Note : Conversion NPU ignorée ou erreur : {e}")