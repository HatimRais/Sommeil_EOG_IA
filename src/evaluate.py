import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from metrics import STAGE_NAMES, compute_metrics, print_metrics, save_metrics


def evaluate_model(model_path, X_test, y_test, save_plot=None):
    """Évalue un modèle Keras : F1 macro, κ, rapport par stade, matrice de confusion."""
    model = tf.keras.models.load_model(model_path)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    metrics = compute_metrics(y_test, y_pred)
    print_metrics(metrics, title=os.path.basename(model_path))
    print(classification_report(
        y_test, y_pred,
        target_names=STAGE_NAMES,
        zero_division=0,
    ))

    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES, cmap="Blues",
    )
    plt.xlabel("Prédiction IA")
    plt.ylabel("Vérité terrain")
    plt.title("Matrice de confusion")
    if save_plot:
        plt.savefig(save_plot, dpi=150, bbox_inches="tight")
        print(f"Figure : {save_plot}")
    else:
        plt.show()
    plt.close()
    return metrics


if __name__ == "__main__":
    # À lancer une fois que train.py a fini
    # Vous devrez charger une partie de vos données X, y ici pour tester
    pass