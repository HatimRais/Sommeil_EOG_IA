"""Métriques de classification (F1, κ, rapport par stade)."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Retourne accuracy, F1 macro/weighted, κ et rapport par classe."""
    labels = labels if labels is not None else list(range(len(STAGE_NAMES)))
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[STAGE_NAMES[i] for i in labels],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "per_class": {
            STAGE_NAMES[i]: {
                "precision": report[STAGE_NAMES[i]]["precision"],
                "recall": report[STAGE_NAMES[i]]["recall"],
                "f1": report[STAGE_NAMES[i]]["f1-score"],
                "support": int(report[STAGE_NAMES[i]]["support"]),
            }
            for i in labels
        },
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(metrics: Dict[str, Any], title: str = "Évaluation") -> None:
    print(f"\n=== {title} ===")
    print(f"  Échantillons : {metrics['n_samples']}")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  F1 macro     : {metrics['f1_macro']:.4f}")
    print(f"  F1 weighted  : {metrics['f1_weighted']:.4f}")
    print(f"  Cohen kappa  : {metrics['cohen_kappa']:.4f}")
    print("  Par stade (F1 / recall / support) :")
    for name, row in metrics["per_class"].items():
        print(f"    {name:3s}  F1={row['f1']:.3f}  recall={row['recall']:.3f}  n={row['support']}")


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
