# DeepSleep AI — Frontend Next.js

Export statique (`output: "export"`) servi par FastAPI en production.

## Dev local (hot reload)

```powershell
# Terminal 1
uvicorn src.api.main:app --reload --port 8000

# Terminal 2
cp .env.local.example .env.local
npm install
npm run dev
```

## Production (Railway)

Le build `npm run build` produit `frontend/out/`, copié dans l'image Docker.
FastAPI sert `/` et `/api/*` sur la même URL — voir [`DEPLOY.md`](../DEPLOY.md).
