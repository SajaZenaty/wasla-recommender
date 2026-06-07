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

# Runtime memory/startup tuning. Set AFTER the build-time model download so the
# download above can still reach Hugging Face.
#   TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE: skip network checks at startup (the
#     model is already baked into the image) to cut memory and time.
#   OMP_NUM_THREADS=1: avoid spawning per-core OpenMP thread pools (torch/faiss/
#     scipy), which reduces peak memory under tight container limits.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=1

COPY . .

# Snapshot dir — mount a Railway Volume at /data in the service settings.
RUN mkdir -p /data \
    && chmod +x scripts/entrypoint.sh

CMD ["scripts/entrypoint.sh"]
