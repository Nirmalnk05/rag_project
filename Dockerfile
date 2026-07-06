FROM python:3.12-slim AS deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
WORKDIR /app

# Bake model weights into the image for air-gapped deploys.
# Build with:  docker build --build-arg PREFETCH_MODELS=true -t rag-app .
# (needs internet AT BUILD TIME; the resulting image runs fully offline).
ARG PREFETCH_MODELS=false
ENV HF_HOME=/app/.hf_cache

RUN useradd -m raguser
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY src/ src/
COPY scripts/ scripts/
COPY data/docs/ data/docs/
RUN mkdir -p data "$HF_HOME" && chown -R raguser:raguser /app
USER raguser

RUN if [ "$PREFETCH_MODELS" = "true" ]; then \
        python scripts/prefetch_models.py; \
    fi

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
