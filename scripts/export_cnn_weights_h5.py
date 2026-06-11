"""Exporte sleep_model_cnn_best.keras -> .weights.h5 pour Kaggle (Keras 3)."""
import os
import sys

import tensorflow as tf

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
keras_path = os.path.join(base, "models", "sleep_model_cnn_best.keras")
h5_path = os.path.join(base, "models", "sleep_model_cnn_best.weights.h5")

if not os.path.isfile(keras_path):
    sys.exit(f"Manquant : {keras_path}")

model = tf.keras.models.load_model(keras_path)
model.save_weights(h5_path)
print(f"OK : {h5_path} ({os.path.getsize(h5_path) / 1e6:.1f} MB)")
