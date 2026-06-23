FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2

# System deps occasionally needed by faiss / scientific wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as UID 1000. Create that user so the
# model cache and snapshot dir are writable at runtime on every platform.
RUN useradd -m -u 1000 user

# Install Python deps as root into the global site-packages.
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Pre-download the embedding model (as the runtime user) so the cache is owned
# by it, the first request is fast, and no network is needed at startup.
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
    OMP_NUM_THREADS=1 \
    USE_MOCK_DATA=true \
    ENABLE_SCHEDULER=false \
    BOOTSTRAP_ON_START=true \
    MOCK_N_USERS=10

# Default snapshot dir (writable by the runtime user). Override with a mounted
# volume / env var on platforms that offer persistent storage.
ENV INDEX_SNAPSHOT_PATH=/home/user/data/index_snapshot.pkl
RUN mkdir -p /home/user/data

COPY --chown=user . .
RUN chmod +x scripts/entrypoint.sh

# Hugging Face Spaces routes to port 7860 by default; Railway injects its own
# PORT which overrides this at runtime.
ENV PORT=7860
EXPOSE 7860

CMD ["scripts/entrypoint.sh"]
