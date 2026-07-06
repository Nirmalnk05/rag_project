# Operations

## Metrics

`GET /metrics` exposes Prometheus text format (dependency-free):
`rag_requests_total`, `rag_cache_hits_total`, `rag_tool_calls_total{tool=...}`,
`rag_answer_latency_seconds_{count,sum}`. Scrape it from Prometheus and build
dashboards/alerts (latency SLOs, error rate, cache-hit ratio). For richer metrics
swap the shim in [src/metrics.py](../../src/metrics.py) for `prometheus_client` —
the `inc()`/`observe()` call sites don't change.

## Prompt versioning

Prompts live in the registry ([src/prompts.py](../../src/prompts.py)) as named,
versioned strings. Add a new version (`"v2"`) alongside `"v1"`, set
`PROMPT_VERSION=v2`, and roll back by flipping it — no code change. This is the
seam for Langfuse Prompt Management or a prompt DB.

## Embedding migration (re-index on model change)

Ingestion records the embedding model in `data/index_meta.json`. If you change
`EMBED_MODEL`, the next ingest **warns** that existing vectors are stale (they were
built with the old model). Re-embed everything:

```bash
python scripts/reembed.py         # == scripts/ingest.py --rebuild
```

## Backup & restore

```bash
python scripts/backup_index.py [backup_dir]     # copies data/*.json + Qdrant snapshot
python scripts/restore_index.py <backup_dir>    # restores artifacts; then reembed if needed
```

The chunk store (`data/chunks.json`) plus the Qdrant snapshot are sufficient to
reconstruct the index; if only the chunk store survives, `scripts/reembed.py`
rebuilds the vectors from it.

## Queue-based ingestion (large corpora)

Ingestion is a blocking batch job today ([scripts/ingest.py](../../scripts/ingest.py)).
For large corpora or continuous updates, run it as a worker job instead of inline:
`run_ingestion(...)` is the unit of work — wrap it in a Celery / Arq / RQ / Temporal
task and trigger on a schedule or on source-system change events (CDC). The
incremental manifest + content-hash caches already make repeated runs cheap and
resumable, so a periodic worker only pays for new/changed documents.

## Health & scaling

`GET /health` (liveness) and `GET /ready` (readiness — 503 until models load) drive
k8s probes. For multi-replica set `RATE_LIMIT_BACKEND=redis`. See
[cloud.md](cloud.md) and [on-prem-airgapped.md](on-prem-airgapped.md).
