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

# 2. Chargement automatique de TOUS les fichiers
print("--- Phase de chargement des données multi-sujets ---")
psg_files = sorted(glob.glob(os.path.join(raw_dir, "*Signal.edf")))
hyp_files = sorted(glob.glob(os.path.join(raw_dir, "*Labels.edf")))

all_X = []
all_y = []

for p, h in zip(psg_files, hyp_files):
    print(f"Traitement du sujet : {os.path.basename(p)}")
    try:
        X_sub, y_sub = create_dataset(p, h)

        # --- OPTIMISATION : NORMALISATION Z-SCORE ---
        # On normalise chaque sujet individuellement pour uniformiser les capteurs
        X_sub = (X_sub - np.mean(X_sub)) / np.std(X_sub)

        all_X.append(X_sub)
        all_y.append(y_sub)
    except Exception as e:
        print(f"Erreur sur le fichier {p}: {e}")

# Fusion
X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)

# Reshape pour le CNN (Epoques, Points, Canal)
X = X.reshape(X.shape[0], X.shape[2], 1).astype(np.float32)

print(f"\nTotal : {X.shape[0]} époques chargées.")
print(f"Distribution : {np.bincount(y)}")

# 3. Équilibrage des classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(weights))

# 4. Création et Entraînement
print("\n--- Initialisation de l'IA (CNN-LSTM) ---")
# On s'assure que l'input shape correspond aux points du signal (ex: 3000 pour 100Hz)
model = build_cnn_lstm_model(input_shape=(X.shape[1], 1))

print("\n--- Début de l'entraînement ---")
history = model.fit(
    X, y,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    shuffle=True
)

# 5. Sauvegarde au format standard
model_save_path = os.path.join(models_dir, "sleep_model_v1.keras")
model.save(model_save_path)
print(f"\nModèle sauvegardé : {model_save_path}")

# 6. OPTIMISATION NPU (OpenVINO FP16)
print("\n--- Conversion stratégique pour Intel AI Boost ---")
try:
    import openvino as ov

    # Dossier temporaire pour l'export TensorFlow
    export_path = os.path.join(models_dir, "temp_export")
    model.export(export_path)

    # Conversion avec compression FP16 (Indispensable pour le NPU)
    ov_model = ov.convert_model(export_path)
    ov_xml_path = os.path.join(models_dir, "sleep_model_npu.xml")

    # On force la compression FP16 ici
    ov.save_model(ov_model, ov_xml_path, compress_to_fp16=True)

    print(f"✅ SUCCÈS : Le modèle est prêt pour le NPU en FP16 !")
    print(f"Fichier : {ov_xml_path}")
except Exception as e:
    print(f"❌ Erreur lors de la conversion NPU : {e}")