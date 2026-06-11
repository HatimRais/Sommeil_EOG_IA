"""
Chargement des modèles .keras exportés depuis Kaggle (Keras 3) avec TensorFlow 2.15 local.
"""
from __future__ import annotations

import io
import zipfile

import h5py
import tensorflow as tf

from architecture import build_cnn_npu_model
from data_loader import N_SAMPLES_PER_EPOCH


def _load_keras3_weights(model: tf.keras.Model, keras_path: str) -> int:
    with zipfile.ZipFile(keras_path) as zf:
        h5_bytes = zf.read("model.weights.h5")
    loaded = 0
    with h5py.File(io.BytesIO(h5_bytes), "r") as f:
        for layer in model.layers:
            weights = []
            i = 0
            while True:
                key = f"layers/{layer.name}/vars/{i}"
                if key not in f:
                    break
                weights.append(f[key][()])
                i += 1
            if weights:
                layer.set_weights(weights)
                loaded += 1
    return loaded


def load_cnn_npu_keras(
    keras_path: str,
    input_shape: tuple[int, int] = (N_SAMPLES_PER_EPOCH, 1),
) -> tf.keras.Model:
    """Charge sleep_model_cnn*.keras (Keras 3 ou local)."""
    try:
        return tf.keras.models.load_model(keras_path)
    except (TypeError, ValueError, OSError):
        model = build_cnn_npu_model(input_shape=input_shape)
        n = _load_keras3_weights(model, keras_path)
        if n == 0:
            raise RuntimeError(
                f"Aucun poids chargé depuis {keras_path} (format Keras 3 attendu)."
            )
        return model
