# DeepSleep AI — Frontend Next.js

Interface web du projet **Sommeil_EOG_IA** (classification des stades du sommeil par EOG).

## Développement local

```bash
cp .env.local.example .env.local
npm install
npm run dev
```

Le backend FastAPI doit tourner sur le port 8000 (voir README à la racine).

## Déploiement Vercel

1. Importer le dépôt GitHub sur [vercel.com/new](https://vercel.com/new)
2. **Root Directory** → `frontend` (obligatoire, monorepo)
3. **Environment Variables** → `NEXT_PUBLIC_API_URL` = URL HTTPS de votre API FastAPI
4. Deploy

> Le backend Python (OpenVINO + MNE) ne peut pas tourner sur Vercel. Hébergez-le séparément (Railway, Render, VM…) et configurez CORS côté API (`CORS_ORIGINS` ou regex `*.vercel.app` déjà activée).

Voir le README racine pour le guide complet.
