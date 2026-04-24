# 🌙 Sommeil_EOG_IA — Classification automatique des stades du sommeil par IA

Pipeline complet de **scoring polysomnographique** à partir d'un signal **EOG (électro-oculographique)**.
Le projet entraîne deux variantes de modèle (**CNN + Bi-LSTM** pour la précision maximale, **CNN 1D pur** pour l'accélération matérielle), les convertit en **OpenVINO IR FP16**, et fournit un **dashboard Streamlit** clinique.

Les 5 stades reconnus suivent la nomenclature AASM : `W` (éveil), `N1`, `N2`, `N3` (sommeil profond, N3+N4 fusionnés) et `REM`.

### ⚡ Accélération matérielle

Le dashboard détecte automatiquement les devices OpenVINO disponibles et permet de choisir entre :

| Device | Modèle | Mesure réelle (Intel Core Ultra 5 125U) |
|---|---|---|
| **🧠 NPU** (Intel AI Boost) | CNN 1D | ~ **5 650 époques/s** (×5.2 vs CPU) |
| 🎮 GPU (iGPU) | CNN 1D | ~ 2 150 époques/s |
| 💻 CPU | CNN 1D ou CNN+LSTM | ~ 1 100 époques/s |

> Le NPU 3720 (Meteor Lake) ne supporte pas les opérations récurrentes (`Loop`, `ReverseSequence`).
> C'est pourquoi un modèle **CNN 1D pur** (`build_cnn_npu_model`) a été développé en parallèle —
> il atteint **88.4 % d'accuracy** sur 4 patients et tourne nativement sur le NPU.

---

## 🧱 Architecture du projet

```
Sommeil_EOG_IA/
│
├── app/
│   └── dashboard.py          # UI Streamlit : upload EDF + inférence OpenVINO + hypnogramme
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Chargement EDF + synchro annotations → (X, y)
│   ├── preprocessing.py      # Resample 100 Hz, filtre FIR 0.5-35 Hz, z-score + clip
│   ├── architecture.py       # 2 modèles : CNN+Bi-LSTM (CPU) et CNN 1D pur (NPU)
│   ├── train.py              # Entraînement variante CNN+Bi-LSTM
│   ├── train_npu.py          # Entraînement variante CNN 1D + export IR FP16 statique
│   └── evaluate.py           # Rapport de classification + matrice de confusion
│
├── data/
│   └── raw/                  # Fichiers EDF bruts  (Patient_XX_Signal.edf / _Labels.edf)
│
├── models/
│   ├── sleep_model_v1.keras  # CNN+Bi-LSTM Keras (CPU only — meilleur F1)
│   ├── sleep_model_cnn.keras # CNN 1D pur Keras (source du modèle NPU)
│   ├── sleep_model_npu.xml   # OpenVINO IR FP16 — shape statique [64, 3000, 1]
│   └── sleep_model_npu.bin   # Poids FP16 (NPU/GPU/CPU compatible)
│
├── .streamlit/
│   └── config.toml          # Thème clinique de base (overridé via CSS dynamique)
│
├── requirements.txt          # Dépendances dashboard + inférence (Cloud)
├── requirements-train.txt    # Optionnel : TensorFlow / entraînement (local)
├── runtime.txt               # Version Python pour Streamlit Cloud
├── .gitignore
└── README.md
```

---

## 🔄 Pipeline de données

```
 EDF brut (100 Hz ou plus)
        │
        ▼
┌────────────────────┐
│ data_loader.py     │   MNE : lecture EDF + annotations hypnogramme
│  create_dataset()  │   → découpage en époques de 30 s (3000 points)
└────────┬───────────┘
         ▼
┌────────────────────┐
│ preprocessing.py   │   Resample 100 Hz · Filtre FIR 0.5-35 Hz
│ clean_eog_signal() │   Normalisation z-score + clip ±3σ
└────────┬───────────┘
         ▼
┌────────────────────┐
│ architecture.py    │   CNN-LSTM :
│ build_cnn_lstm…()  │   Conv1D(64) → Conv1D(128) → BiLSTM(64) → Dense(5)
└────────┬───────────┘
         ▼
┌────────────────────┐
│ train.py           │   Entraînement multi-sujets + pondération de classes
│                    │   Export .keras puis conversion OpenVINO FP16
└────────┬───────────┘
         ▼
┌────────────────────┐
│ dashboard.py       │   Streamlit : inférence temps réel via OpenVINO
│                    │   Affichage hypnogramme IA vs vérité-terrain
└────────────────────┘
```

---

## 🧠 Modèles disponibles

Deux architectures co-existent dans `src/architecture.py` :

### A. CNN + Bi-LSTM (CPU only — précision maximale)

| Bloc | Couches | Rôle |
|------|---------|------|
| **CNN** | `Conv1D(64,k=3)` → `BN` → `MaxPool` → `Conv1D(128,k=3)` → `BN` → `MaxPool` | Motifs morphologiques |
| **Bi-LSTM** | `Bidirectional(LSTM(64))` | Dépendances temporelles |
| **Classifieur** | `Dense(64)` → `Dense(5, softmax)` | Probabilités stades |

→ **~ 91.6 % accuracy** sur Patient_01.
→ Incompatible NPU (les ops `Loop` / `ReverseSequence` du Bi-LSTM ne sont pas supportées par le NPU 3720).

### B. CNN 1D pur (NPU-compatible — accélération matérielle)

| Bloc | Couches | Sortie |
|---|---|---|
| 1 | `Conv1D(64, k=11)` → `BN` → `MaxPool(4)` | (750, 64) |
| 2 | `Conv1D(128, k=7)` → `BN` → `MaxPool(4)` | (187, 128) |
| 3 | `Conv1D(256, k=5)` → `BN` → `MaxPool(4)` | (46, 256) |
| 4 | `Conv1D(256, k=3)` → `BN` → `MaxPool(2)` | (23, 256) |
| Tête | `GlobalAvgPool1D` → `Dense(128)` → `Dense(5, softmax)` | (5,) |

- **455 k paramètres** (1.74 MB) · **Entrée statique** `(64, 3000, 1)`
- **88.4 % accuracy** moyenne sur 4 patients · **5 650 ép/s sur NPU** ⚡
- Toutes les ops sont supportées par le NPU 3720 : `Convolution`, `MaxPool`, `BN`, `ReLU`, `ReduceMean`, `MatMul`, `Softmax`

---

## 🚀 Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
# Optionnel — entraînement / évaluation Keras (TensorFlow) :
# pip install -r requirements-train.txt
```

---

## ☁️ Déploiement (Streamlit Community Cloud)

Le projet est prêt pour **[Streamlit Community Cloud](https://streamlit.io/cloud)** : le dashboard n’utilise que **OpenVINO + MNE + NumPy/Pandas** (`requirements.txt`), sans TensorFlow à l’exécution.

### Prérequis

| Élément | Détail |
|--------|--------|
| Dépôt | **GitHub** (public pour l’offre Community) avec le code et les poids |
| Fichier principal | `app/dashboard.py` (à indiquer dans les paramètres de l’app) |
| Dépendances | `requirements.txt` à la racine (détecté automatiquement) |
| Python | **3.12 recommandé** : `runtime.txt` contient `python-3.12`. Sur Community Cloud, le menu **Advanced settings** lors du déploiement peut **ignorer** `runtime.txt` — choisir explicitement **Python 3.12** (éviter **3.14** : pas de wheels pour d’anciennes versions d’OpenVINO, et build `pandas` depuis les sources). |
| Modèles | Les fichiers `models/sleep_model_npu.xml` et `models/sleep_model_npu.bin` doivent être **versionnés** (ou fournis via stockage externe + script de téléchargement). Sans eux, le message *Model file not found* s’affiche. |
| Accélération | **Pas de NPU** sur Cloud : seuls **CPU** (et éventuellement **GPU** selon l’offre) sont disponibles. Choisir **CPU** ou **AUTO** dans la barre latérale. |

### Étapes rapides

1. Pousser le dépôt sur GitHub (inclure `models/` si les binaires ne dépassent pas la limite du dépôt ; au-delà, utiliser [Git LFS](https://git-lfs.com/) ou héberger les poids ailleurs).
2. Sur [share.streamlit.io](https://share.streamlit.io), **New app** → choisir le dépôt, la branche, et **Main file path** : `app/dashboard.py`.
3. Lancer le déploiement ; le premier build peut prendre quelques minutes (téléchargement des wheels, etc.).

Aucun secret n’est requis pour l’application telle quelle. Les réglages globaux par défaut sont dans [`.streamlit/config.toml`](.streamlit/config.toml).

### Problème « Error installing requirements » (OpenVINO / pandas / uv)

Typiquement : l’environnement Cloud utilise **Python 3.14** par défaut → `openvino==2024.0.0` n’a **pas de wheel** pour cette ABI → échec du premier installeur (`uv`), puis `pip` tente de **compiler** `numpy` / `pandas` depuis les sources → erreurs (`pkg_resources`, etc.).

**À faire :**

1. Mettre à jour le dépôt avec le `requirements.txt` actuel (OpenVINO **≥ 2024.4**, `pandas` **≥ 2.1`, `setuptools`).
2. Dans **Manage app** → **Settings** (ou en **redéployant** l’app), ouvrir **Advanced settings** et fixer **Python version** sur **3.12** (ou **3.11**). Si l’interface ne permet pas de changer la version, **supprimer l’app** et la recréer pour choisir la version au moment du déploiement.
3. Enregistrer / relancer le déploiement.

Les fichiers **IR OpenVINO** (`sleep_model_npu.xml` / `.bin`) exportés avec une version antérieure restent en général lisibles par un runtime OpenVINO plus récent (inférence CPU).

---

## 📥 Données

Placer les fichiers **EDF** dans `data/raw/` en respectant la convention :

```
data/raw/Patient_01_Signal.edf     # Signal polysomnographique (contient un canal EOG)
data/raw/Patient_01_Labels.edf     # Hypnogramme annoté par un expert
```

Formats compatibles : **Sleep-EDF Expanded** (PhysioNet) ou tout EDF contenant un canal `EOG …`.

---

## 🏋️ Entraînement

### Variante CNN 1D (NPU)
```bash
python src/train_npu.py
```
Sorties :
- `models/sleep_model_cnn.keras`
- `models/sleep_model_npu.xml` + `.bin` (IR FP16, batch statique = 64)
- Test automatique de compilation sur tous les devices détectés en fin de run.

### Variante CNN + Bi-LSTM (CPU)
```bash
python src/train.py
```
Sortie : `models/sleep_model_v1.keras`

---

## 📊 Évaluation

```python
from src.evaluate import evaluate_model
evaluate_model("models/sleep_model_v1.keras", X_test, y_test)
```

Produit :
- Le `classification_report` par classe (précision / rappel / F1)
- Une matrice de confusion (heatmap seaborn)

---

## 🖥️ Dashboard

```bash
streamlit run app/dashboard.py
```

Fonctionnalités :
- **Sélecteur de thème** : `System` (suit l'OS) · `Light` · `Dark`
- **Sélecteur de device matériel** : NPU / GPU / CPU / AUTO (auto-détecté)
- **Upload** d'un EDF (signal, + optionnel labels pour validation)
- ou sélection d'un **patient local** dans `data/raw/`
- Inférence par batches de 64 (compatible NPU statique)
- 5 onglets : Hypnogram · Sleep Architecture · AI vs Expert · Clinical Report · Technical
- 10 KPIs cliniques : TST, Sleep Efficiency, REM%, Deep Sleep%, Sleep Latency, WASO, Awakenings, etc.
- Export CSV des prédictions par époque

### Thèmes

| Mode | Comportement |
|---|---|
| **System** | Suit la préférence de l'OS via `prefers-color-scheme` (clair par défaut, bascule en sombre si l'OS est en sombre) |
| **Light** | Palette clinique claire — fond gris froid, primaire bleu marine `#1B4965`, accent bleu médian `#2C7DA0` |
| **Dark** | Palette clinique sombre — fond bleu nuit `#0B1220`, primaire bleu pâle `#5FA8D3`, accent turquoise `#62D2C4` |

Les trois thèmes partagent les mêmes composants (cartes patient, métriques, chips de statut, hypnogramme, matrice de confusion). Seules les variables CSS changent — aucun rerun n'est nécessaire pour la cohérence visuelle.

L’interface du dashboard intègre des **media queries** (`@media (max-width: 900px / 768px / 480px)`) : en vue téléphone, les `st.columns` s’empilent en une colonne, l’en-tête clinique se replie, les onglets défilent horizontalement, les zones de toucher ciblent **≥ 44 px** de hauteur, et les champs de saisie utilisent **16 px** de police côté iOS pour limiter le zoom sur focus. Les bords sûrs (`safe-area-inset`) sont pris en compte pour encoches / barre d’accueil.

---

## ⚙️ Pile technologique

| Domaine | Outils |
|---|---|
| Signal biomédical | `MNE-Python` |
| Deep Learning | `TensorFlow / Keras` |
| Inférence optimisée | `OpenVINO` (Intel AI Boost / NPU) |
| ML classique | `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn` |
| UI | `Streamlit` |

---

## 📄 Licence

Projet académique — usage pédagogique et de recherche.
