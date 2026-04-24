"""
Entraînement de la variante CNN-only et export OpenVINO IR FP16 NPU-compatible.

Sortie :
  - models/sleep_model_cnn.keras       (modèle Keras)
  - models/sleep_model_npu.xml / .bin  (IR FP16 statique batch=BATCH_INFER)
"""
import os
import glob
import shutil
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import create_dataset
from architecture import build_cnn_npu_model

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
BATCH_TRAIN = 64
EPOCHS = 15
BATCH_INFER = 64          # batch fixe utilisé pour l'IR NPU
N_POINTS = 3000

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir = os.path.join(base_path, "data", "raw")
models_dir = os.path.join(base_path, "models")
os.makedirs(models_dir, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Chargement multi-sujets
# ----------------------------------------------------------------------------
print("--- Chargement des données multi-sujets ---")
psg_files = sorted(glob.glob(os.path.join(raw_dir, "*Signal.edf")))
hyp_files = sorted(glob.glob(os.path.join(raw_dir, "*Labels.edf")))

all_X, all_y = [], []
for p, h in zip(psg_files, hyp_files):
    print(f"  Sujet : {os.path.basename(p)}")
    try:
        X_sub, y_sub = create_dataset(p, h)
        X_sub = (X_sub - np.mean(X_sub)) / (np.std(X_sub) + 1e-6)
        all_X.append(X_sub)
        all_y.append(y_sub)
    except Exception as e:
        print(f"    ERREUR : {e}")

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
X = X.reshape(X.shape[0], X.shape[2], 1).astype(np.float32)
print(f"\nTotal : {X.shape[0]} époques | Distribution : {np.bincount(y)}")

# Mélange GLOBAL avant le split (sinon validation = 1 patient entier non vu).
rng = np.random.default_rng(seed=42)
perm = rng.permutation(X.shape[0])
X = X[perm]
y = y[perm]

# ----------------------------------------------------------------------------
# 2. Pondération des classes
# ----------------------------------------------------------------------------
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(weights))
print(f"Pondération : {class_weights}")

# ----------------------------------------------------------------------------
# 3. Modèle CNN-only NPU-compatible
# ----------------------------------------------------------------------------
print("\n--- Construction du modèle CNN (NPU-compatible) ---")
model = build_cnn_npu_model(input_shape=(N_POINTS, 1))
model.summary()

print("\n--- Entraînement ---")
history = model.fit(
    X, y,
    epochs=EPOCHS,
    batch_size=BATCH_TRAIN,
    validation_split=0.2,
    class_weight=class_weights,
    shuffle=True,
    verbose=2,
)

# ----------------------------------------------------------------------------
# 4. Sauvegarde Keras
# ----------------------------------------------------------------------------
keras_path = os.path.join(models_dir, "sleep_model_cnn.keras")
model.save(keras_path)
print(f"\n[Keras] sauvegardé : {keras_path}")

# ----------------------------------------------------------------------------
# 5. Export OpenVINO IR FP16 — shape statique batch fixe pour le NPU
# ----------------------------------------------------------------------------
print(f"\n--- Export OpenVINO IR (batch statique = {BATCH_INFER}) ---")
import openvino as ov

export_dir = os.path.join(models_dir, "_tmp_saved_model")
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
model.export(export_dir)

ov_model = ov.convert_model(export_dir, input=[(BATCH_INFER, N_POINTS, 1)])
# Force static shape (NPU 3720 ne supporte que les shapes statiques).
ov_model.reshape({ov_model.inputs[0]: [BATCH_INFER, N_POINTS, 1]})
xml_path = os.path.join(models_dir, "sleep_model_npu.xml")
ov.save_model(ov_model, xml_path, compress_to_fp16=True)
print(f"[OpenVINO] sauvegardé : {xml_path}  (batch statique = {BATCH_INFER})")

shutil.rmtree(export_dir, ignore_errors=True)

# ----------------------------------------------------------------------------
# 6. Test immédiat sur NPU / GPU / CPU
# ----------------------------------------------------------------------------
print("\n--- Test de compilation sur tous les devices ---")
core = ov.Core()
print(f"Devices disponibles : {core.available_devices}")

X_test = np.random.randn(BATCH_INFER, N_POINTS, 1).astype(np.float32)
for dev in core.available_devices:
    try:
        m = core.read_model(xml_path)
        net = core.compile_model(m, dev)
        out = net(X_test)[net.output(0)]
        exec_dev = list(net.get_property("EXECUTION_DEVICES"))
        print(f"  [{dev}] OK  out={out.shape}  exec={exec_dev}")
    except Exception as e:
        print(f"  [{dev}] FAIL : {str(e).splitlines()[0][:120]}")

print("\nTermine.")
