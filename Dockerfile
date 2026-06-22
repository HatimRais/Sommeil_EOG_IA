# DeepSleep AI — Next.js + FastAPI + OpenVINO (Railway, modèle SICAM)
# ── Stage 1 : build frontend ──────────────────────────────────────────────
FROM node:20-bookworm-slim AS frontend

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ── Stage 2 : runtime Python + Node ───────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements-prod.txt requirements-api.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-prod.txt -r requirements-api.txt

COPY src/ src/
COPY models/sleep_model_npu.xml models/sleep_model_npu.bin models/
COPY data/raw/ data/raw/
COPY scripts/railway_entrypoint.py scripts/verify_inference_models.py scripts/

# Next.js standalone
COPY --from=frontend /app/frontend/.next/standalone ./frontend/
COPY --from=frontend /app/frontend/.next/static ./frontend/.next/static
COPY --from=frontend /app/frontend/public ./frontend/public

ENV NODE_ENV=production \
    HOSTNAME=0.0.0.0 \
    DEEPSLEEP_INTERNAL_API_PORT=8001 \
    DEEPSLEEP_INFERENCE_URL=http://127.0.0.1:8001 \
    PORT=3000

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=3 \
    CMD python -c "import os,urllib.request; p=os.environ.get('PORT','3000'); urllib.request.urlopen(f'http://127.0.0.1:{p}/api/health', timeout=10)"

CMD ["python", "scripts/railway_entrypoint.py"]
