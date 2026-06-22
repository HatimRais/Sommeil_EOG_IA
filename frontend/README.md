# DeepSleep AI — Frontend Next.js

Interface web avec **proxy API intégré** (modèle SICAM).

## Local

```bash
# Terminal 1 (racine)
uvicorn src.api.main:app --reload --port 8000

# Terminal 2
cp .env.local.example .env.local
npm install
npm run dev
```

Les routes `src/app/api/*` proxy vers FastAPI sur le port 8000.

## Production (Railway)

Un seul service Docker à la racine du repo — voir [`DEPLOY.md`](../DEPLOY.md).

Aucune variable d'environnement obligatoire sur Railway.
