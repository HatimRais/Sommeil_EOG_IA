"""
Évaluation sur le jeu **test** (hold-out jury) — à lancer APRÈS train_npu.py.

Usage :
  python src/evaluate_holdout.py
  python src/evaluate_holdout.py --model models/sleep_model_cnn_best.keras
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from keras_compat import load_cnn_npu_keras
from metrics import compute_metrics, print_metrics, save_metrics
from splits import load_split_manifest

BATCH = 64


def load_corpus_from_cache(base_path: str):
    cache = os.path.join(base_path, "data", "processed", "sleep_edf_corpus.npz")
    meta_path = os.path.join(base_path, "data", "processed", "sleep_edf_corpus_meta.json")
    if not os.path.isfile(cache):
        raise FileNotFoundError(
            f"Cache introuvable : {cache}\nLancez d'abord : python src/train_npu.py"
        )
    data = np.load(cache)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return data["X"], data["y"], data["subject_idx"], meta["subject_names"]


def main():
    parser = argparse.ArgumentParser(description="Évaluation hold-out (jury)")
    parser.add_argument(
        "--model",
        default=None,
        help="Chemin .keras (défaut : sleep_model_cnn_best.keras puis sleep_model_cnn.keras)",
    )
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split_path = os.path.join(base, "data", "processed", "subject_split.json")
    models_dir = os.path.join(base, "models")

    if not os.path.isfile(split_path):
        raise FileNotFoundError(
            f"Split introuvable : {split_path}\nEntraînez d'abord avec train_npu.py."
        )

    manifest = load_split_manifest(split_path)
    test_names = set(manifest["test_subjects"])
    print(f"Hold-out jury : {len(test_names)} sujets")
    for name in manifest["test_subjects"][:8]:
        print(f"  - {name}")
    if len(test_names) > 8:
        print(f"  ... +{len(test_names) - 8} autres (voir subject_split.json)")

    X, y, subject_idx, subject_names = load_corpus_from_cache(base)
    test_ids = {i for i, name in enumerate(subject_names) if name in test_names}
    test_mask = np.isin(subject_idx, list(test_ids))
    if not test_mask.any():
        raise RuntimeError("Aucune époque test dans le cache.")

    X_test, y_test = X[test_mask], y[test_mask]
    print(f"\nÉpoques test : {len(y_test)} (jamais vues à l'entraînement)")

    model_path = args.model
    if not model_path:
        for cand in ("sleep_model_cnn_best.keras", "sleep_model_cnn.keras"):
            p = os.path.join(models_dir, cand)
            if os.path.isfile(p):
                model_path = p
                break
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError("Modèle Keras introuvable dans models/")

    print(f"Modèle : {model_path}")
    model = load_cnn_npu_keras(model_path)
    y_pred = np.argmax(model.predict(X_test, batch_size=BATCH, verbose=0), axis=1)

    metrics = compute_metrics(y_test, y_pred)
    print_metrics(metrics, title="TEST hold-out (jury)")

    out = os.path.join(models_dir, "metrics_holdout_test.json")
    save_metrics(metrics, out)
    print(f"\nRapport jury : {out}")
    print(f"Liste sujets démo : {split_path}")


if __name__ == "__main__":
    main()
