"""
Entraînement CNN 1D (NPU) sur Sleep-EDF Expanded + export OpenVINO.

Split par **sujet** en 3 parties :
  - train   (~70 %) — apprentissage
  - val     (~15 %) — early stopping / F1 macro (jamais mélangé avec test)
  - test    (~15 %) — hold-out **jury** : non utilisé ici → evaluate_holdout.py

Usage :
  python src/train_npu.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils import class_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from architecture import build_cnn_npu_model
from data_loader import N_SAMPLES_PER_EPOCH, load_corpus
from metrics import compute_metrics, print_metrics, save_metrics
from splits import (
    DEFAULT_JURY_DEMO_SUBJECTS,
    DEFAULT_SEED,
    DEFAULT_TEST_FRAC,
    DEFAULT_VAL_FRAC,
    save_split_manifest,
    subject_train_val_test_split,
)

BATCH_TRAIN = 64
BATCH_INFER = 64
EPOCHS = 30
PATIENCE = 7

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sleep_edf_root = os.path.join(base_path, "data", "sleep-edf-database-expanded-1.0.0")
legacy_raw_dir = os.path.join(base_path, "data", "raw")
processed_dir = os.path.join(base_path, "data", "processed")
models_dir = os.path.join(base_path, "models")
cache_path = os.path.join(processed_dir, "sleep_edf_corpus.npz")
cache_meta_path = os.path.join(processed_dir, "sleep_edf_corpus_meta.json")
split_path = os.path.join(processed_dir, "subject_split.json")

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


class F1MacroCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_f1 = -1.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        macro = float(f1_score(self.y_val, y_pred, average="macro", zero_division=0))
        logs["val_f1_macro"] = macro
        print(f"  -> val_f1_macro = {macro:.4f}", flush=True)
        if macro > self.best_f1 + 1e-4:
            self.best_f1 = macro
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= PATIENCE:
            print(f"  Early stop (F1 macro plateau, patience={PATIENCE})", flush=True)
            self.model.stop_training = True


def load_or_build_corpus():
    if os.path.isfile(cache_path) and os.path.isfile(cache_meta_path):
        print(f"--- Chargement cache : {cache_path} ---", flush=True)
        data = np.load(cache_path)
        with open(cache_meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return data["X"], data["y"], data["subject_idx"], meta["subject_names"]

    print("--- Construction du corpus (Sleep-EDF + raw) — peut prendre ~30 min ---", flush=True)
    X, y, subject_idx, subject_names = load_corpus(
        sleep_edf_root=sleep_edf_root if os.path.isdir(sleep_edf_root) else None,
        legacy_raw_dir=legacy_raw_dir if os.path.isdir(legacy_raw_dir) else None,
    )
    np.savez_compressed(cache_path, X=X, y=y, subject_idx=subject_idx)
    with open(cache_meta_path, "w", encoding="utf-8") as f:
        json.dump({"subject_names": subject_names, "n_epochs": int(len(y))}, f, indent=2)
    print(f"Cache enregistré : {cache_path}", flush=True)
    return X, y, subject_idx, subject_names


def main():
    tf.random.set_seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)

    X, y, subject_idx, subject_names = load_or_build_corpus()
    n_samples = X.shape[1]
    if n_samples != N_SAMPLES_PER_EPOCH:
        print(f"ATTENTION : {n_samples} points/époque (attendu {N_SAMPLES_PER_EPOCH})", flush=True)

    # --- Split 3 volets (test = hold-out jury, exclu de fit) ---
    force_test = [
        n for n in DEFAULT_JURY_DEMO_SUBJECTS
        if n in subject_names
    ]
    train_mask, val_mask, test_mask, manifest = subject_train_val_test_split(
        subject_idx,
        subject_names,
        test_frac=DEFAULT_TEST_FRAC,
        val_frac=DEFAULT_VAL_FRAC,
        seed=DEFAULT_SEED,
        force_test_names=force_test,
    )
    save_split_manifest(manifest, split_path)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    # X_test, y_test : réservés — évaluation via evaluate_holdout.py

    print(f"\n=== Split par sujet (hold-out jury) ===", flush=True)
    print(f"  Train : {manifest['n_subjects_train']} sujets · {manifest['n_epochs_train']} époques", flush=True)
    print(f"  Val   : {manifest['n_subjects_val']} sujets · {manifest['n_epochs_val']} époques", flush=True)
    print(f"  Test  : {manifest['n_subjects_test']} sujets · {manifest['n_epochs_test']} époques  [NON UTILISÉ à l'entraînement]", flush=True)
    print(f"  Manifeste : {split_path}", flush=True)
    if manifest["test_subjects"][:5]:
        print(f"  Ex. sujets jury : {', '.join(manifest['test_subjects'][:5])} ...", flush=True)
    print(f"Train dist : {np.bincount(y_train, minlength=5)}", flush=True)
    print(f"Val dist   : {np.bincount(y_val, minlength=5)}", flush=True)

    weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(weights))

    print("\n--- Modèle CNN (NPU-compatible) ---", flush=True)
    model = build_cnn_npu_model(input_shape=(n_samples, 1))
    model.summary()

    best_keras = os.path.join(models_dir, "sleep_model_cnn_best.keras")
    callbacks = [
        F1MacroCallback(X_val, y_val),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            best_keras, monitor="val_f1_macro", mode="max",
            save_best_only=True, verbose=1,
        ),
    ]

    print("\n--- Entraînement (train + val uniquement) ---", flush=True)
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_TRAIN,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    if os.path.isfile(best_keras):
        model = tf.keras.models.load_model(best_keras)

    y_pred_val = np.argmax(model.predict(X_val, batch_size=BATCH_TRAIN, verbose=0), axis=1)
    metrics_val = compute_metrics(y_val, y_pred_val)
    print_metrics(metrics_val, title="Validation (sujets val — pas le hold-out jury)")

    save_metrics(metrics_val, os.path.join(models_dir, "metrics_val.json"))
    model.save(os.path.join(models_dir, "sleep_model_cnn.keras"))

    print(f"\n--- Export OpenVINO (batch={BATCH_INFER}) ---", flush=True)
    import openvino as ov

    export_dir = os.path.join(models_dir, "_tmp_saved_model")
    shutil.rmtree(export_dir, ignore_errors=True)
    model.export(export_dir)
    ov_model = ov.convert_model(export_dir, input=[(BATCH_INFER, n_samples, 1)])
    ov_model.reshape({ov_model.inputs[0]: [BATCH_INFER, n_samples, 1]})
    xml_path = os.path.join(models_dir, "sleep_model_npu.xml")
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)
    shutil.rmtree(export_dir, ignore_errors=True)

    print(f"\nTerminé.", flush=True)
    print(f"  → Évaluation jury (test hold-out) : python src/evaluate_holdout.py", flush=True)
    print(f"  → Liste sujets démo : {split_path}", flush=True)


if __name__ == "__main__":
    main()
