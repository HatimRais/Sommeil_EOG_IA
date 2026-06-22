# Déploiement DeepSleep AI — Railway (tout-en-un)

**Une seule URL** : FastAPI sert l'API (`/api/*`) **et** le frontend Next.js (`/`).

```
Navigateur → uvicorn :PORT
              ├── GET /           → frontend/out/index.html
              ├── GET /api/health → JSON API
              └── POST /api/analyze → inférence EOG
```

Le build compile le frontend en fichiers statiques (`frontend/out`) inclus dans l'image.

---

## Étape 1 — Pousser sur GitHub

```powershell
git push origin main
```

Fichiers requis dans Git : `models/sleep_model_npu.*`, `data/raw/*.edf`

---

## Étape 2 — Déployer sur Railway

1. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
2. Repo **Sommeil_EOG_IA** — **Root Directory = racine** (vide)
3. **Settings → Build** :
   - Builder : **Dockerfile** (recommandé) ou laisser Railpack + `nixpacks.toml`
   - **Start Command** : *(vide — utiliser le CMD du Dockerfile)*
4. **Settings → Resources** : RAM **2 Go**
5. **Networking → Generate Domain**

### Variables d'environnement

**Aucune obligatoire.**

| Variable | Quand |
|----------|-------|
| `PORT` | Injecté automatiquement par Railway |

> ⚠️ Si vous voyez `{"detail":"Not Found"}` sur `/` : le frontend n'a pas été buildé.
> Vérifiez les logs de **build** (pas seulement deploy) — cherchez `npm run build`.
> Supprimez toute **Start Command** custom du type `uvicorn ...` sans build frontend.

---

## Étape 3 — Vérifier

| Test | Résultat attendu |
|------|------------------|
| `https://VOTRE-URL/` | Interface DeepSleep AI (HTML) |
| `https://VOTRE-URL/api/health` | `{"status":"ok","service":"DeepSleep AI"}` |
| Analyse Patient_01 | Hypnogramme affiché |

Logs de démarrage corrects :
```
INFO: Uvicorn running on http://0.0.0.0:8080
INFO: "GET / HTTP/1.1" 200 OK
INFO: "GET /api/health HTTP/1.1" 200 OK
```

---

## Dev local

```powershell
# Terminal 1
pip install -r requirements-prod.txt -r requirements-api.txt
cd frontend && npm run build && cd ..
uvicorn src.api.main:app --reload --port 8000

# Ouvrir http://localhost:8000
```

Ou dev frontend séparé :
```powershell
# Terminal 1 : uvicorn port 8000
# Terminal 2 : cd frontend && cp .env.local.example .env.local && npm run dev
```

---

## Dépannage

| Symptôme | Cause | Solution |
|----------|-------|----------|
| `{"detail":"Not Found"}` sur `/` | API seule, pas de `frontend/out` | Builder Dockerfile ou nixpacks avec `npm run build` |
| Logs : uvicorn seul, pas de build npm | Railpack sans phase build | Utiliser Dockerfile ou `nixpacks.toml` |
| Start Command custom | Écrase le Dockerfile CMD | Vider Start Command dans Settings |
| OOM | RAM insuffisante | 2 Go minimum |
| 504 | Mauvais port exposé | Domaine sur PORT public (8080), pas un port interne |
