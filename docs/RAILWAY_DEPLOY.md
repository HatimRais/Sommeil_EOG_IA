# Déploiement Railway — DeepSleep AI

Guide pas à pas pour héberger **tout le projet** sur [Railway](https://railway.app) : 2 services dans un même projet (API Python + frontend Next.js).

---

## Architecture sur Railway

```
┌─────────────────────────────────────────────────────────┐
│  Projet Railway : Sommeil_EOG_IA                        │
│                                                         │
│  ┌─────────────────────┐    ┌─────────────────────────┐ │
│  │  deepsleep-api      │    │  deepsleep-web          │ │
│  │  (Dockerfile racine)│◄───│  (frontend/Dockerfile)  │ │
│  │  FastAPI + OpenVINO │    │  Next.js 16             │ │
│  └──────────┬──────────┘    └────────────┬────────────┘ │
│             │                            │              │
│     *.up.railway.app              *.up.railway.app      │
└─────────────────────────────────────────────────────────┘
         ▲                                    │
         └──── NEXT_PUBLIC_API_URL ───────────┘
              (navigateur → API publique HTTPS)
```

> **Ordre obligatoire** : déployer l’**API en premier**, récupérer son URL, puis déployer le **frontend** avec cette URL.

---

## Checklist Railway (à suivre dans l’ordre)

### Étape 0 — Prérequis

- [ ] Compte [Railway](https://railway.app) créé
- [ ] Dépôt GitHub `Sommeil_EOG_IA` poussé sur `main` (avec `Dockerfile`, `frontend/Dockerfile`, modèles OpenVINO, `data/raw/`)
- [ ] Plan Railway actif (carte requise au-delà du crédit gratuit)

---

### Étape 1 — Créer le projet

1. [railway.app/new](https://railway.app/new) → **Deploy from GitHub repo**
2. Sélectionner **`Sommeil_EOG_IA`**
3. Railway crée un premier service automatiquement → on le configurera comme **API**

---

### Étape 2 — Service API (`deepsleep-api`) — **EN PREMIER**

Ouvrir le service → **Settings** :

| Paramètre | Valeur |
|---|---|
| **Service name** | `deepsleep-api` |
| **Root Directory** | *(vide — racine du repo)* |
| **Builder** | Dockerfile |
| **Dockerfile path** | `Dockerfile` |
| **Config file** | `railway.toml` *(auto)* |

**Settings → Resources** :

| Paramètre | Valeur recommandée |
|---|---|
| **Memory** | **2 Go** minimum (OpenVINO + MNE + EDF ~50 Mo) |
| **CPU** | 2 vCPU si disponible |

**Settings → Networking** :

- [ ] Cliquer **Generate Domain**
- [ ] Copier l’URL publique, ex. :  
  `https://deepsleep-api-production.up.railway.app`
- [ ] Tester dans le navigateur : `https://VOTRE-URL/api/health`  
  → doit retourner `{"status":"ok","service":"DeepSleep AI"}`

**Variables d’environnement (service API)** :

| Variable | Valeur | Obligatoire ? | Quand la définir |
|---|---|---|---|
| `PORT` | *(injecté automatiquement par Railway)* | Auto | — |
| `CORS_ORIGINS` | URL du frontend (étape 4) | Oui | **Après** déploiement du frontend |

> Ne pas définir `CORS_ORIGINS` tout de suite — vous n’avez pas encore l’URL du frontend.

**Deploy** → attendre le build Docker (~3–8 min la 1ʳᵉ fois).

---

### Étape 3 — Service Frontend (`deepsleep-web`) — **EN SECOND**

Dans le **même projet Railway** :

1. **+ New** → **GitHub Repo** → sélectionner à nouveau `Sommeil_EOG_IA`
2. Configurer le nouveau service :

| Paramètre | Valeur |
|---|---|
| **Service name** | `deepsleep-web` |
| **Root Directory** | `frontend` |
| **Builder** | Dockerfile |
| **Dockerfile path** | `Dockerfile` |
| **Config file** | `frontend/railway.toml` *(auto)* |

**Variables d’environnement (service Frontend)** — **AVANT le premier deploy** :

| Variable | Valeur | Obligatoire ? | Notes |
|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | `https://deepsleep-api-production.up.railway.app` | **Oui** | URL API **sans** `/` final — copier depuis l’étape 2 |
| `PORT` | *(auto Railway)* | Auto | — |

> ⚠️ `NEXT_PUBLIC_*` est compilé **au build**. Si vous changez l’URL API plus tard, **redéployez** le frontend (Rebuild).

**Settings → Networking** :

- [ ] **Generate Domain**
- [ ] Copier l’URL, ex. :  
  `https://deepsleep-web-production.up.railway.app`

**Deploy** → attendre le build (~2–5 min).

---

### Étape 4 — Relier CORS (API ← Frontend)

Retourner sur le service **`deepsleep-api`** → **Variables** :

| Variable | Valeur |
|---|---|
| `CORS_ORIGINS` | `https://deepsleep-web-production.up.railway.app` |

*(Remplacer par votre URL frontend réelle. Pas de `/` final.)*

- [ ] **Redéployer** ou **Restart** le service API pour appliquer la variable.

> Les domaines `*.up.railway.app` sont aussi autorisés par regex côté code, mais `CORS_ORIGINS` reste utile pour un **domaine personnalisé**.

---

### Étape 5 — Vérification finale

- [ ] Ouvrir l’URL frontend → badge **« API connectée »** (vert)
- [ ] Sélectionner un patient (`Patient_01` … `Patient_12`) → lancer l’analyse
- [ ] Hypnogramme + métriques s’affichent (~10–30 s selon la taille EDF)
- [ ] Export CSV → colonnes séparées dans Excel FR
- [ ] Upload d’un fichier EDF externe fonctionne

**En cas de « API hors ligne »** :

1. Vérifier `NEXT_PUBLIC_API_URL` sur le frontend (URL exacte de l’API)
2. Redéployer le frontend si la variable a été modifiée
3. Tester `https://VOTRE-API/api/health` directement
4. Consulter les **Logs** des deux services sur Railway

---

## Récapitulatif des variables

### Service `deepsleep-api`

| Variable | Exemple | Rôle |
|---|---|---|
| `PORT` | `8080` | Port interne (auto Railway) |
| `CORS_ORIGINS` | `https://deepsleep-web-production.up.railway.app` | Autorise le frontend (domaine custom) |

### Service `deepsleep-web`

| Variable | Exemple | Rôle |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `https://deepsleep-api-production.up.railway.app` | URL de l’API pour le navigateur |
| `PORT` | `3000` | Port interne (auto Railway) |

---

## Domaine personnalisé (optionnel)

Pour chaque service → **Settings → Networking → Custom Domain** :

| Service | Sous-domaine suggéré |
|---|---|
| API | `api.votredomaine.com` |
| Frontend | `app.votredomaine.com` ou `votredomaine.com` |

Puis mettre à jour :

- Frontend : `NEXT_PUBLIC_API_URL=https://api.votredomaine.com` → **redéployer**
- API : `CORS_ORIGINS=https://app.votredomaine.com` → **restart**

---

## Coûts et limites

| Élément | Détail |
|---|---|
| **Image Docker API** | ~200 Mo de données EDF + deps Python |
| **RAM API** | 2 Go recommandés (pic à l’inférence) |
| **NPU/GPU** | Non disponible sur Railway → device **CPU** ou **AUTO** |
| **Upload EDF** | Fichiers jusqu’à ~50 Mo supportés |
| **Cold start** | 1ʳᵉ requête après inactivité peut prendre 30–60 s (chargement OpenVINO) |

---

## Commandes locales (test avant deploy)

```bash
# Build API
docker build -t deepsleep-api .
docker run -p 8000:8000 deepsleep-api

# Build frontend
cd frontend
docker build --build-arg NEXT_PUBLIC_API_URL=http://localhost:8000 -t deepsleep-web .
docker run -p 3000:3000 -e PORT=3000 deepsleep-web
```

---

## Fichiers ajoutés pour Railway

| Fichier | Rôle |
|---|---|
| `Dockerfile` | Image API (Python 3.12 + OpenVINO) |
| `railway.toml` | Config service API |
| `requirements-prod.txt` | Dépendances prod sans Streamlit |
| `.dockerignore` | Exclut notebooks, frontend, corpus 8 Go |
| `frontend/Dockerfile` | Image Next.js standalone |
| `frontend/railway.toml` | Config service frontend |
