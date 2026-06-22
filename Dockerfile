# DeepSleep AI — FastAPI + frontend statique (export Next.js)
# ── Stage 1 : build frontend ──────────────────────────────────────────────
FROM node:20-bookworm-slim AS frontend

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ── Stage 2 : runtime Python ────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements-prod.txt requirements-api.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-prod.txt -r requirements-api.txt

COPY src/ src/
COPY models/sleep_model_npu.xml models/sleep_model_npu.bin models/
COPY data/raw/ data/raw/
COPY --from=frontend /app/frontend/out ./frontend/out

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=3 \
    CMD python -c "import os,urllib.request; p=os.environ.get('PORT','8080'); urllib.request.urlopen(f'http://127.0.0.1:{p}/api/health', timeout=10)"

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
