# Déploiement DeepSleep AI — Railway (tout-en-un, modèle SICAM)

Une seule URL Railway : interface Next.js + API FastAPI interne + modèles OpenVINO **dans l'image Docker**.

```
Navigateur → Next.js (port public PORT) → proxy /api/* → FastAPI 127.0.0.1:8001
```

> **Note :** le projet SICAM utilise le même schéma sur Railway (pas Vercel pour l'API).
> Vercel ne convient pas à OpenVINO + MNE + fichiers EDF ~50 Mo.

---

## Étape 1 — Vérifier les fichiers versionnés

Ces fichiers doivent être dans Git (déjà inclus) :

```
models/sleep_model_npu.xml
models/sleep_model_npu.bin
data/raw/Patient_*_Signal.edf
data/raw/Patient_*_Labels.edf
```

```powershell
git status
git push origin main
```

---

## Étape 2 — Déployer sur Railway

1. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
2. Repo **Sommeil_EOG_IA** (racine du dépôt)
3. Railway utilise le `Dockerfile` automatiquement
4. **Settings → Networking → Generate Domain**

| Port | Rôle | Exposé sur Internet ? |
|------|------|------------------------|
| `PORT` (ex. 8080) | Site Next.js | **Oui** ← domaine Railway |
| 8001 | API FastAPI interne | **Non** |

> ⚠️ Ne pas exposer le port **8001** — vous obtiendriez un **504**.

### Ressources recommandées

| Ressource | Minimum |
|-----------|---------|
| RAM | **2 Go** (OpenVINO + MNE + EDF) |
| CPU | 1–2 vCPU |

### Variables d'environnement

**Aucune variable obligatoire.** Railway injecte `PORT` automatiquement.

| Variable | Défaut | Rôle |
|----------|--------|------|
| `PORT` | auto | Port public Next.js |
| `DEEPSLEEP_INTERNAL_API_PORT` | `8001` | Port API interne |
| `DEEPSLEEP_INFERENCE_URL` | `http://127.0.0.1:8001` | URL pour les routes proxy Next.js |

---

## Étape 3 — Vérifier

1. Ouvrez `https://VOTRE-URL.up.railway.app`
2. Badge **« API connectée »** (vert)
3. Lancez une analyse sur `Patient_01`
4. Export CSV → colonnes séparées dans Excel FR

```powershell
curl https://VOTRE-URL.up.railway.app/api/health
```

Réponse attendue : `{"status":"ok","service":"DeepSleep AI"}`

---

## Développement local

```powershell
# Terminal 1 — API FastAPI
pip install -r requirements-prod.txt -r requirements-api.txt
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Next.js (proxy vers 8000)
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Les routes `frontend/src/app/api/*` proxy vers `http://127.0.0.1:8000` — même schéma qu'en production.

---

## Dépannage

| Symptôme | Solution |
|----------|----------|
| Build échoue `COPY models/` | Vérifiez que `sleep_model_npu.xml/.bin` sont commités |
| Crash « Modèles introuvables » | Logs build : modèles absents de l'image |
| OOM | Augmentez la RAM à 2 Go |
| Healthcheck timeout | Premier boot ~2–3 min (OpenVINO + MNE) |
| **504 Bad Gateway** | Domaine pointé sur **8001** au lieu de **PORT** |
| API hors ligne en local | Terminal 1 : uvicorn sur port 8000 |

---

## Architecture (identique à SICAM)

```
┌──────────────────────────────────────────────┐
│  Railway — un seul conteneur Docker          │
│                                              │
│  Browser → Next.js :PORT                     │
│              ↓ /api/health, /api/analyze…    │
│            FastAPI 127.0.0.1:8001            │
│              ↓                               │
│         OpenVINO + MNE + data/raw            │
└──────────────────────────────────────────────┘
```
