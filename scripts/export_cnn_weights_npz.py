"""Exporte les poids CNN NPU en .npz (compatible Kaggle Keras 3)."""
import os
import sys

import numpy as np
import tensorflow as tf

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
keras_path = os.path.join(base, "models", "sleep_model_cnn_best.keras")
npz_path = os.path.join(base, "models", "sleep_model_cnn_best_weights.npz")

if not os.path.isfile(keras_path):
    sys.exit(f"Manquant : {keras_path}")

model = tf.keras.models.load_model(keras_path)
weights = {}
for layer in model.layers:
    w = layer.get_weights()
    if w:
        for i, arr in enumerate(w):
            weights[f"{layer.name}__{i}"] = arr.astype(np.float32)

np.savez_compressed(npz_path, **weights)
print(f"OK : {npz_path} ({os.path.getsize(npz_path) / 1e6:.2f} MB)")
