"""
Découpage par **sujet** : train / validation / test (hold-out jury).

Règle : les sujets du jeu **test** ne sont jamais vus pendant l'entraînement
ni pour l'early stopping. Évaluation finale : evaluate_holdout.py.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

DEFAULT_TEST_FRAC = 0.15   # ~30 sujets / 201 — réservés jury
DEFAULT_VAL_FRAC = 0.15    # early stopping
DEFAULT_SEED = 42

# Sujets locaux (data/raw) forcés dans le hold-out pour démo dashboard jury
DEFAULT_JURY_DEMO_SUBJECTS = ("Patient_01", "Patient_02")


def subject_train_val_test_split(
    subject_idx: np.ndarray,
    subject_names: List[str],
    test_frac: float = DEFAULT_TEST_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    seed: int = DEFAULT_SEED,
    force_test_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Masques booléens train / val / test + manifeste (liste des sujets par fold).
    """
    unique = np.unique(subject_idx)
    n_total = len(unique)
    forced_names = set(force_test_names or [])

    forced_ids = np.array(
        [s for s in unique if subject_names[int(s)] in forced_names], dtype=int
    )
    pool = np.array([s for s in unique if s not in forced_ids], dtype=int)

    n_test_target = max(len(forced_ids), int(round(n_total * test_frac)))
    n_extra = max(0, n_test_target - len(forced_ids))

    if n_extra > 0 and len(pool) > 0:
        # sklearn renvoie (train, test) — ne pas inverser les noms
        pool_trainval, pool_test = train_test_split(
            pool,
            test_size=n_extra,
            random_state=seed,
        )
        test_subj = (
            np.concatenate([forced_ids, pool_test]) if len(forced_ids) else pool_test
        )
        trainval_pool = pool_trainval
    elif len(forced_ids):
        test_subj = forced_ids
        trainval_pool = pool if len(pool) else unique
    else:
        trainval_pool, test_subj = train_test_split(
            unique, test_size=test_frac, random_state=seed
        )

    val_rel = val_frac / max(1e-9, 1.0 - len(test_subj) / n_total)
    val_rel = float(np.clip(val_rel, 0.05, 0.4))
    train_subj, val_subj = train_test_split(
        trainval_pool, test_size=val_rel, random_state=seed + 1
    )

    train_mask = np.isin(subject_idx, train_subj)
    val_mask = np.isin(subject_idx, val_subj)
    test_mask = np.isin(subject_idx, test_subj)

    def _names_for_subjects(subj_ids: np.ndarray) -> List[str]:
        ids = np.unique(subject_idx[subj_ids]) if subj_ids.any() else np.array([], dtype=int)
        return sorted(subject_names[int(i)] for i in ids)

    manifest = {
        "seed": seed,
        "test_frac": test_frac,
        "val_frac": val_frac,
        "n_subjects_total": n_total,
        "n_subjects_train": int(len(np.unique(subject_idx[train_mask]))),
        "n_subjects_val": int(len(np.unique(subject_idx[val_mask]))),
        "n_subjects_test": int(len(np.unique(subject_idx[test_mask]))),
        "n_epochs_train": int(train_mask.sum()),
        "n_epochs_val": int(val_mask.sum()),
        "n_epochs_test": int(test_mask.sum()),
        "train_subjects": _names_for_subjects(train_mask),
        "val_subjects": _names_for_subjects(val_mask),
        "test_subjects": _names_for_subjects(test_mask),
        "forced_test_subjects": sorted(forced_names),
        "note": (
            "test_subjects = hold-out jury. Ne jamais entraîner dessus. "
            "Démo : charger un enregistrement listé dans test_subjects (ex. Patient_01). "
            "Métriques officielles : python src/evaluate_holdout.py"
        ),
    }
    return train_mask, val_mask, test_mask, manifest


def save_split_manifest(manifest: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_split_manifest(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
