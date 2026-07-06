# Air-Gapped On-Prem Deployment

The whole system runs with **zero internet egress**. The only thing that ever
needs the network is the *first-time download of model weights* — bake those in
a connected staging environment, transfer, then run disconnected.

## What runs, and why it's offline

| Component | Offline how |
|---|---|
| LLM | `LLM_PROVIDER=ollama` — models run in the local Ollama container |
| Embeddings + reranker (BGE) | local inference; weights baked into the image |
| Document parsing + OCR (Docling) | local; layout/OCR weights baked in |
| Figure captioning | local vision model via Ollama (or disabled) |
| Vector DB (Qdrant) | self-hosted container |
| BM25 | in-process |
| Rate limiting | in-memory (single node) |
| Tracing | off (or self-hosted Langfuse) |

Nothing calls Anthropic, OpenAI, LlamaParse, Azure, HuggingFace Hub, or cloud
Langfuse in this profile.

## Step 1 — stage in a CONNECTED environment

```bash
# 1a. Build the app image with model weights baked in (needs internet now).
docker build --build-arg PREFETCH_MODELS=true -t rag-app:offline .

# 1b. Pull the LLM + vision models into the Ollama volume.
docker compose -f docker-compose.offline.yml up -d ollama
docker exec rag-ollama ollama pull llama3.1          # tool-capable (agent mode)
docker exec rag-ollama ollama pull llama3.2-vision   # figure captioning (optional)

# 1c. Export the image and the ollama volume for transfer.
docker save rag-app:offline qdrant/qdrant:v1.12.0 ollama/ollama:latest -o rag-images.tar
docker run --rm -v production-rag_ollama_models:/v -v "$PWD":/out alpine \
  tar czf /out/ollama_models.tgz -C /v .
```

## Step 2 — move to the air-gapped host

Transfer `rag-images.tar`, `ollama_models.tgz`, and the repo (for
`docker-compose.offline.yml`, `.env.offline.example`, and `data/docs/`).

## Step 3 — run disconnected

```bash
docker load -i rag-images.tar
# restore the ollama volume
docker volume create production-rag_ollama_models
docker run --rm -v production-rag_ollama_models:/v -v "$PWD":/in alpine \
  tar xzf /in/ollama_models.tgz -C /v

cp .env.offline.example .env          # LLM_PROVIDER=ollama, PARSE_BACKEND=local, HF_*_OFFLINE=1
docker compose -f docker-compose.offline.yml up -d
docker exec rag-app python scripts/ingest.py    # ingest your documents
```

Open the chat UI at `http://<host>:8000/` and confirm an agent query runs with
the network cable pulled.

## Notes

- **Agent mode needs a tool-capable model.** `llama3.1`+ supports tool-calling;
  larger models are noticeably more reliable at it. Use `AGENT_MODE=pipeline` if
  your local model can't do tools.
- **Runtime offline flag.** `.env.offline.example` sets `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` so sentence-transformers/Docling load only from the
  baked cache.
- **Offline eval.** `scripts/run_eval.py` uses RAGAS, which defaults to OpenAI —
  point RAGAS at local models before running eval air-gapped (serving is
  unaffected).
- Put real documents in `data/docs/` (`.pdf`, `.docx`, `.xlsx`, `.md`).
