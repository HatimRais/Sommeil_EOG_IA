# DeepSleep AI — API FastAPI (Railway service 1)
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

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8000') + '/api/health')" || exit 1

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
