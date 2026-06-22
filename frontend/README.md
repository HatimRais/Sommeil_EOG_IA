# DeepSleep AI — Frontend Next.js

Interface web du projet **Sommeil_EOG_IA**.

## Local

```bash
cp .env.local.example .env.local
npm install
npm run dev
```

Backend requis sur le port 8000 (voir README racine).

## Railway

Ce dossier est le **service 2** du déploiement Railway.

| Paramètre Railway | Valeur |
|---|---|
| Root Directory | `frontend` |
| Dockerfile | `Dockerfile` |
| Variable | `NEXT_PUBLIC_API_URL` = URL du service API |

Guide complet : [`docs/RAILWAY_DEPLOY.md`](../docs/RAILWAY_DEPLOY.md)
