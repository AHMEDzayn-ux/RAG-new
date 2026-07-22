FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding models into the image so a cold start never has to reach
# Hugging Face. Fetching them at runtime made KB indexing fail intermittently on
# Cloud Run cold starts ("Can't load the model ... all-MiniLM-L6-v2"), which left
# the FAISS collection un-created and every retrieval dead. We pre-download both
# the multilingual model (config default) and the English MiniLM (embeddings.py
# fallback default) into an in-image HF cache, then run fully offline below.
ENV HF_HOME=/app/hf_cache
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

# Storage paths default to ../documents and ../vector_stores (see config.py) which
# assumes the monorepo layout; point them inside the image instead since this
# Dockerfile ships backend/ as a standalone build context (Hugging Face Spaces).
ENV DOCUMENTS_DIR=/app/documents
ENV VECTOR_STORES_DIR=/app/vector_stores

# Models are baked in above — never hit the HF network at runtime, so cold-start
# KB indexing is deterministic (no rate-limit / egress dependency).
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8080

# Shell form (not exec/JSON form) so $PORT actually expands - Cloud Run injects its
# own PORT (default 8080); HF Spaces/other hosts that don't set it fall back to 8080.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
