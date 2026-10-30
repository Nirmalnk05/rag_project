# Cloud Deployment (Kubernetes, multi-replica)

The API is stateless with respect to *request handling* and scales
horizontally, but each replica is **not** a blank process: it loads the whole
chunk store into RAM and rebuilds the BM25 index at startup
(`Retriever.__init__`, [src/retrieval.py:100-102](../../src/retrieval.py)). So
replica memory scales with corpus size, cold start is corpus-proportional, and
replicas can disagree about the corpus — see
[aws.md §4](aws.md) for the version-skew hazard and
[aws.md §11](aws.md) for the fix. Manifests are in `deploy/k8s/`
(example-grade — adjust image, resources, ingress, and secrets management).

## Topology

```
Ingress/LB → Service → rag-app (N replicas)
                          ├── Qdrant   (managed Qdrant Cloud, or the StatefulSet)
                          └── Redis     (shared rate-limit counter)
LLM: Anthropic API (or self-hosted Ollama/vLLM in-cluster)
```

## Multi-replica correctness

- **Rate limiting must be shared.** Set `RATE_LIMIT_BACKEND=redis` +
  `REDIS_URL`; otherwise each replica counts independently and the effective
  limit is `rpm × replicas`. The default `memory` backend is single-instance only.
- **Readiness probe.** `/ready` returns 503 until models + orchestrator finish
  loading; `/health` is liveness. Both are wired in `app-deployment.yaml`. Cold
  start loads the embedding + reranker models, so the readiness `initialDelay` /
  `failureThreshold` are generous.
- **Shared index.** Ingestion is a batch job that writes `data/chunks.json`; every
  replica rebuilds its in-process BM25 from it at startup and shares the same
  Qdrant. Use a ReadWriteMany PVC for `data/`, or run ingestion as a Job that
  publishes the artifacts all replicas mount.
- **Ingestion-time caches** (`context_cache.json`, `caption_cache.json`) are used
  only during the batch ingest job, not at serving time — they don't need sharing.

## Deploy

```bash
# 1. Build + push the image.
docker build -t your-registry/rag-app:latest .
docker push your-registry/rag-app:latest
# (edit the image in deploy/k8s/app-deployment.yaml)

# 2. Config + secrets (prefer a real secrets manager over the example Secret).
kubectl apply -f deploy/k8s/configmap.example.yaml
kubectl apply -f deploy/k8s/secret.example.yaml

# 3. Dependencies. Managed Qdrant Cloud is recommended; else the StatefulSet.
kubectl apply -f deploy/k8s/redis.yaml
kubectl apply -f deploy/k8s/qdrant-statefulset.yaml   # skip if using managed Qdrant

# 4. App.
kubectl apply -f deploy/k8s/app-deployment.yaml
kubectl apply -f deploy/k8s/app-service.yaml

# 5. Verify scaling + shared limit.
kubectl scale deploy/rag-app --replicas=3
kubectl get pods -l app=rag-app
# hammer the API past RATE_LIMIT_RPM and confirm 429s appear regardless of which
# pod serves (proves the shared redis counter).
```

## Scaling notes

- GPU nodes help embeddings/reranker/Docling and any in-cluster local LLM. With a
  cloud LLM (Anthropic), app pods are CPU-bound and cheap to scale.
- Front with an Ingress + TLS; set `API_KEYS` to require `X-API-Key`.
- For HA Qdrant use replicas + snapshots/backups (managed Qdrant handles this).
