import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import os


def evaluate_model(model_path, X_test, y_test):
    # 1. Charger le modèle sauvegardé
    model = tf.keras.models.load_model(model_path)

    # 2. Faire les prédictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # 3. Générer le rapport de classification
    target_names = ['Eveil (W)', 'N1', 'N2', 'N3', 'REM']
    print("\n--- Rapport de Classification ---")
    print(classification_report(y_test, y_pred_classes, target_names=target_names))

    # 4. Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.xlabel('Prédictions de l\'IA')
    plt.ylabel('Vérité Terrain (Expert)')
    plt.title('Matrice de Confusion : Analyse du Sommeil EOG')
    plt.show()


if __name__ == "__main__":
    # À lancer une fois que train.py a fini
    # Vous devrez charger une partie de vos données X, y ici pour tester
    pass