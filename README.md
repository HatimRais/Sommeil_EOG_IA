# 🌙 Sommeil_EOG_IA — Classification automatique des stades du sommeil par IA

Pipeline complet de **scoring polysomnographique** à partir d'un signal **EOG (électro-oculographique)**.
Le projet entraîne deux variantes de modèle (**CNN + Bi-LSTM** pour la précision maximale, **CNN 1D pur** pour l'accélération matérielle), exporte la variante CNN en **OpenVINO IR FP16**, et fournit un **dashboard Streamlit** clinique (**DeepSleep AI**).

Les 5 stades reconnus suivent la nomenclature AASM : `W` (éveil), `N1`, `N2`, `N3` (sommeil profond, N3+N4 fusionnés) et `REM`.

### ⚡ Accélération matérielle

Le dashboard détecte automatiquement les devices OpenVINO disponibles et permet de choisir entre :

| Device | Modèle déployé | Mesure indicative (Intel Core Ultra 5 125U) |
|---|---|---|
| **🧠 NPU** (Intel AI Boost) | CNN 1D (IR OpenVINO) | ~ **5 650 époques/s** (×5.2 vs CPU) |
| 🎮 GPU (iGPU) | CNN 1D | ~ 2 150 époques/s |
| 💻 CPU | CNN 1D (IR) ou CNN+LSTM (.keras) | ~ 1 100 époques/s |

> Le NPU 3720 (Meteor Lake) ne supporte pas les opérations récurrentes (`Loop`, `ReverseSequence`).
> C'est pourquoi un modèle **CNN 1D pur** (`build_cnn_npu_model`) a été développé en parallèle —
> il atteint **~88.4 % d'accuracy** sur 4 patients et tourne nativement sur le NPU via `sleep_model_npu.xml`.

Le **dashboard** charge uniquement le modèle **OpenVINO** (`models/sleep_model_npu.xml`). Les fichiers `.keras` servent à l'entraînement et à l'évaluation hors ligne.

---

## 🧱 Architecture du projet

```
Sommeil_EOG_IA/
│
├── app/
│   ├── dashboard.py          # UI Streamlit (legacy)
│   └── ui_theme.py           # Tokens CSS — source du thème Next.js
│
├── frontend/                 # Interface Next.js (React 19 · Tailwind)
│   └── src/                  # Composants, graphiques Recharts, API client
│
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI REST (backend Next.js)
│   │   └── service.py        # Logique d'analyse partagée
│   ├── data_loader.py        # create_dataset() (entraînement) · load_and_sync_labels() (dashboard)
│   ├── preprocessing.py      # apply_preprocessing() — pipeline inférence dashboard (100 Hz)
│   ├── architecture.py       # build_cnn_lstm_model() · build_cnn_npu_model()
│   ├── train.py              # Entraînement CNN+Bi-LSTM → sleep_model_v1.keras
│   ├── train_npu.py          # Entraînement CNN 1D + export OpenVINO IR FP16
│   └── evaluate.py           # Rapport sklearn + matrice de confusion (modèle Keras)
│
├── scripts/
│   └── generate_rapport_prof_docx.py   # Génère Rapport_Projet_Sommeil_EOG_IA.docx
│
├── data/raw/                 # Patient_XX_Signal.edf · Patient_XX_Labels.edf (4 sujets fournis)
│
├── models/
│   ├── sleep_model_v1.keras      # CNN+Bi-LSTM (CPU)
│   ├── sleep_model_cnn.keras     # CNN 1D (source avant export OV)
│   ├── sleep_model_npu.xml       # OpenVINO IR — entrée statique [64, 3000, 1]
│   └── sleep_model_npu.bin
│
├── .streamlit/config.toml    # Thème Streamlit de base (complété par ui_theme.py)
├── requirements.txt          # Dashboard + inférence (sans TensorFlow)
├── requirements-api.txt      # FastAPI + uvicorn (backend Next.js)
├── requirements-train.txt      # TensorFlow — entraînement local uniquement
├── runtime.txt               # python-3.12 (Streamlit Cloud)
├── Rapport_Projet_Sommeil_EOG_IA.docx   # Synthèse projet (optionnel)
└── README.md
```

---

## 🔄 Pipelines de données

Deux chemins coexistent selon l'usage :

### A. Entraînement (`train.py` / `train_npu.py`)

```
*_Signal.edf + *_Labels.edf
        │
        ▼
  create_dataset()          # data_loader.py — MNE
        │                   Canal « EOG horizontal », filtre 0.3–35 Hz
        │                   Époques 30 s · labels AASM (event_id MNE)
        ▼
  Z-score par patient · fusion multi-sujets
        │
        ├── train.py        → sleep_model_v1.keras      (CNN + Bi-LSTM)
        └── train_npu.py    → sleep_model_cnn.keras
                            → sleep_model_npu.xml/.bin  (export OpenVINO FP16)
```

### B. Inférence dashboard (`app/dashboard.py`)

```
*_Signal.edf  (upload ou data/raw/)
        │
        ▼
  apply_preprocessing()     # preprocessing.py
        │                   Resample 100 Hz · canal EOG (nom contenant « EOG »)
        │                   Filtre 0.5–35 Hz · z-score · clip ±3σ
        ▼
  Découpage époques 30 s → tenseur (N, 3000, 1)
        │
        ▼
  OpenVINO (sleep_model_npu.xml) — batch 64
        │
        ├── Hypnogramme · métriques cliniques · export CSV
        └── Optionnel : *_Labels.edf → load_and_sync_labels() + parse_stage()
```

> **Note :** le prétraitement d'entraînement (`create_dataset`) et celui du dashboard (`apply_preprocessing`) ne sont pas strictement identiques (canal fixe vs détection EOG, bandes passantes légèrement différentes). L'inférence suit le pipeline du dashboard.

---

## 🧠 Modèles disponibles

Deux architectures dans `src/architecture.py` :

### A. CNN + Bi-LSTM (CPU — `sleep_model_v1.keras`)

| Bloc | Couches | Rôle |
|------|---------|------|
| **CNN** | `Conv1D(64,k=3)` → `BN` → `MaxPool` → `Conv1D(128,k=3)` → `BN` → `MaxPool` | Motifs locaux |
| **Bi-LSTM** | `Bidirectional(LSTM(64))` | Contexte temporel |
| **Classifieur** | `Dense(64)` → `Dense(5, softmax)` | 5 stades AASM |

→ **~91.6 % accuracy** sur Patient_01 (indicatif).  
→ Incompatible NPU (ops `Loop` / `ReverseSequence`).

### B. CNN 1D pur (NPU — `sleep_model_npu.xml`)

| Bloc | Couches | Sortie indicative |
|---|---|---|
| 1 | `Conv1D(64, k=11)` → `BN` → `MaxPool(4)` | (750, 64) |
| 2 | `Conv1D(128, k=7)` → `BN` → `MaxPool(4)` | (187, 128) |
| 3 | `Conv1D(256, k=5)` → `BN` → `MaxPool(4)` | (46, 256) |
| 4 | `Conv1D(256, k=3)` → `BN` → `MaxPool(2)` | (23, 256) |
| Tête | `GlobalAvgPool1D` → `Dense(128)` → `Dense(5, softmax)` | (5,) |

- **~455 k paramètres** · entrée statique **`(64, 3000, 1)`** (batch 64 × 30 s × 100 Hz)
- **~88.4 % accuracy** moyenne sur 4 patients · **~5 650 ép/s sur NPU**
- Ops supportées NPU : `Convolution`, `MaxPool`, `BN`, `ReLU`, `ReduceMean`, `MatMul`, `Softmax`

---

## 🚀 Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
# Entraînement / évaluation Keras (local uniquement) :
# pip install -r requirements-train.txt
```

Lancer le dashboard :

```bash
streamlit run app/dashboard.py
```

---

## ☁️ Déploiement (Streamlit Community Cloud)

Le dashboard n'utilise que **OpenVINO + MNE + NumPy/Pandas/Streamlit** (`requirements.txt`), sans TensorFlow à l'exécution.

| Élément | Détail |
|--------|--------|
| Fichier principal | `app/dashboard.py` |
| Python | **3.12** — `runtime.txt` ; choisir **3.12** dans Advanced settings (éviter 3.14) |
| Modèles | `models/sleep_model_npu.xml` et `.bin` doivent être présents dans le dépôt (ou LFS / URL) |
| Accélération | **Pas de NPU** sur Cloud → **CPU** ou **AUTO** |

### Problèmes fréquents Cloud

- **OpenVINO / pandas / Python 3.14** : utiliser `requirements.txt` actuel (`openvino>=2024.4`, `pandas>=2.1`, `setuptools`) et **Python 3.12**.
- **`sph_harm` / SciPy** : MNE 1.6 exige **`scipy>=1.11,<1.14`** (déjà borné dans `requirements.txt`).

---

## 📥 Données

Convention dans `data/raw/` :

```
data/raw/Patient_01_Signal.edf     # Signal PSG (canal EOG pour l'entraînement : « EOG horizontal »)
data/raw/Patient_01_Labels.edf     # Hypnogramme expert (annotations)
```

Jeu fourni : **Patient_01, 02, 11, 12**. Formats compatibles : **Sleep-EDF Expanded** (PhysioNet) ou tout EDF avec canal EOG et annotations de stades.

---

## 🏋️ Entraînement

Dataset **Sleep-EDF Expanded** (PhysioNet) : placer l’archive décompressée dans  
`data/sleep-edf-database-expanded-1.0.0/` (sous-dossiers `sleep-cassette/` et `sleep-telemetry/`).  
Les 4 fichiers `data/raw/` restent inclus automatiquement (**201 sujets** au total).

Depuis la **racine** du projet (TensorFlow requis : `pip install -r requirements-train.txt`) :

### Variante CNN 1D (NPU + export OpenVINO)

```bash
python src/train_npu.py
```

- **1ᵉʳ run** : charge ~197 enregistrements Sleep-EDF + `raw/` (~30 min), met en cache `data/processed/sleep_edf_corpus.npz`
- **Split par sujet en 3** (≈70 % train · 15 % val · 15 % **test hold-out jury**) — le jeu **test** n’est **jamais** utilisé à l’entraînement
- Manifeste : `data/processed/subject_split.json` (liste des sujets pour la démo devant le jury)
- Métrique principale : **F1 macro** sur validation ; rapport jury : `python src/evaluate_holdout.py` → `models/metrics_holdout_test.json`
- Sorties : `sleep_model_cnn.keras`, `sleep_model_cnn_best.keras`, `sleep_model_npu.xml` / `.bin`, `models/metrics_val.json`

### Variante CNN + Bi-LSTM (CPU)

```bash
python src/train.py
```

Sortie : `models/sleep_model_v1.keras` (pas d'export OpenVINO automatique).

Les deux scripts chargent tous les couples `*Signal.edf` / `*Labels.edf` via `create_dataset()` et appliquent une **pondération de classes** (`class_weight`) pour le déséquilibre des stades.

---

## 📊 Évaluation

```python
from src.evaluate import evaluate_model
evaluate_model("models/sleep_model_v1.keras", X_test, y_test)
```

Produit un `classification_report` et une matrice de confusion (matplotlib / seaborn).

---

## 🖥️ Interface Next.js (recommandée)

Interface moderne **React / Next.js** avec thème médical, graphiques interactifs et mode clair/sombre.

### 1. Backend API (FastAPI)

```bash
pip install -r requirements.txt -r requirements-api.txt
uvicorn src.api.main:app --reload --port 8000
```

Ou sous Windows : `.\scripts\run_api.ps1`

### 2. Frontend Next.js

```bash
cd frontend
cp .env.local.example .env.local   # ou copier manuellement
npm install
npm run dev
```

Ouvrir [http://localhost:3000](http://localhost:3000)

| Composant | Technologie |
|---|---|
| UI | Next.js 16 · React 19 · Tailwind CSS 4 |
| Thème | next-themes (clair / sombre / système) — tokens `ui_theme.py` |
| Graphiques | Recharts (hypnogramme, architecture, matrice) |
| API | FastAPI · OpenVINO · MNE |

---

## 🖥️ Dashboard Streamlit (legacy)

```bash
streamlit run app/dashboard.py
```

Fonctionnalités :

- **Thème** : System · Light · Dark (`app/ui_theme.py`, variables `--dps-*`)
- **Matériel** : NPU / GPU / CPU / AUTO (OpenVINO)
- **Données** : upload EDF signal (+ labels optionnels) ou base `data/raw/`
- **Inférence** : batches de 64 · modèle `sleep_model_npu.xml`
- **Onglets** : Hypnogram · Sleep Architecture · AI vs Expert · Clinical Report · Technical
- **KPIs** : TST, efficacité du sommeil, latences, % stades, éveils, cycles, etc.
- **Export** : CSV époque par époque

### Thèmes (palette `ui_theme.py`)

| Mode | Comportement |
|---|---|
| **System** | Clair par défaut ; `@media (prefers-color-scheme: dark)` bascule le thème sombre |
| **Light** | Fond `#EEF1F5`, primaire `#0C4A6E`, accent `#1E5F8A` |
| **Dark** | Fond `#0B1020`, primaire `#5BA3D0`, accent `#5EB8A8` |

Couleurs des **stades** (hypnogramme / légende) : fixes AASM — W `#EAB308`, N1 `#93C5E8`, N2 `#3B82B6`, N3 `#0C4A6E`, REM `#B91C1C`.

Interface **responsive** (breakpoints 900 / 768 / 480 px) : colonnes empilées sur mobile, onglets scrollables, zones tactiles ≥ 44 px.

---

## 📄 Rapport de synthèse (optionnel)

Générer ou régénérer le document Word à la racine :

```bash
pip install python-docx   # si absent
python scripts/generate_rapport_prof_docx.py
```

→ `Rapport_Projet_Sommeil_EOG_IA.docx`

---

## ⚙️ Pile technologique

| Domaine | Outils |
|---|---|
| Signal biomédical | MNE-Python 1.6 |
| Deep Learning (entraînement) | TensorFlow / Keras (`requirements-train.txt`) |
| Inférence optimisée | OpenVINO ≥ 2024.4 (NPU / GPU / CPU) |
| ML classique | scikit-learn |
| Visualisation | matplotlib |
| UI | Streamlit |

---

## 📄 Licence

Projet académique — usage pédagogique et de recherche.
