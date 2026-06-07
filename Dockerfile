FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2

WORKDIR /app

# System deps occasionally needed by faiss / scientific wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding model so the first request is not slow and the
# container does not need network access at startup.
RUN python -c "from sentence_transformers import SentenceTransformer; \
import os; SentenceTransformer(os.environ['EMBEDDING_MODEL_NAME'])"

COPY . .

# Snapshot persistence volume.
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=3).raise_for_status()"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
