# AWS Deployment — Design Proposal

> **Status: design proposal, not a verified runbook.** Nothing here has been
> executed against a live AWS account. Service names, IAM shapes and pricing
> must be re-checked against current AWS docs before you build. The generic
> Kubernetes guide in [cloud.md](cloud.md) remains the source of truth for
> manifest-level detail; this document is the AWS-specific binding and the
> reasoning behind it.

## 1. Guiding decision: one architecture, swapped backing services

We deploy **the same container and the same application architecture** on-prem
and in AWS, and swap only the *backing services* behind the existing
environment switches. We do **not** fork the design.

This is viable because the codebase already has the right seams:

| Seam | Env var | On-prem | AWS |
|---|---|---|---|
| LLM backend | `LLM_PROVIDER` | `ollama` | `bedrock` (new) or `openai_compatible` via proxy |
| Session memory | `MEMORY_BACKEND` | `memory` | `redis` |
| Rate limiting | `RATE_LIMIT_BACKEND` | `memory` | `redis` |
| Parsing | `PARSE_BACKEND` | `local` | `local` (unchanged) |
| Guardrails | `GUARDRAILS_BACKEND` | `heuristic` | `heuristic` |

**What must stay identical in both environments:** retrieval logic, chunking,
prompts (`PROMPT_VERSION`) and the eval suite. If those drift, a quality
regression found in one environment will not reproduce in the other, and the
RAGAS/CI gates stop being meaningful.

**What is allowed to differ:** compute platform, replica counts, storage,
secrets delivery, identity provider, LLM backend.

## 2. Service mapping

| Concern | AWS service | Notes |
|---|---|---|
| Compute | **ECS Fargate** | Decided — see §3. On-prem is Docker Compose, so EKS's parity argument does not apply |
| Vector store | **Qdrant Cloud (AWS region)** | Strongly preferred over self-hosting; removes the HA/backup/upgrade burden flagged as an open gap |
| Shared state | **ElastiCache for Redis** | Rate-limit counter, session memory, and (once implemented) the shared semantic cache |
| LLM | **Amazon Bedrock** (Anthropic models) | Requires explicit model access enablement per account/region |
| Embeddings / rerank | In-process `sentence-transformers` | BGE-small + bge-reranker-base. CPU is adequate at low QPS — see §6 |
| Index artifacts | **S3** | Versioned `chunks.json`, manifest, `index_meta`; target for `backup_index.py` |
| Ingestion | **ECS task** (scheduled or S3 → EventBridge triggered) | `scripts/ingest.py` unchanged. Not Lambda — Docling/OCR dependency size and runtime exceed its limits |
| Identity | **ALB + Cognito (OIDC)** | Replaces today's spoofable header ACL — see §5 |
| Secrets | **Secrets Manager** / SSM Parameter Store | Injected as env vars; never baked into the image |
| Metrics / logs | **AMP** + **AMG**, CloudWatch Logs | `/metrics` is already Prometheus-format |
| Tracing | Self-hosted Langfuse *or* disabled | See §7 — data residency caveat |

## 3. Compute: ECS Fargate (decided)

**On-prem runs Docker Compose, and no Kubernetes rollout is planned** — for
either multiple users at a single customer site or redeployment to separate
customers. That settles this choice.

EKS's one strong argument here was reusing `deploy/k8s/` for genuine
on-prem/cloud parity. With no on-prem Kubernetes, that argument disappears
entirely, leaving only its costs: a per-cluster control-plane charge, cluster
upgrades, and add-on/IRSA/CNI operations — for a system whose corpus ceiling is
roughly 1k–5k documents and which needs single-digit replica counts.

**Therefore: ECS Fargate.** It matches the system's actual scale and keeps
operational surface small.

`deploy/k8s/` stays in the repo as a maintained *optional* path, for two cases:

1. A customer mandates deployment into their own existing Kubernetes cluster
   (a procurement constraint, not an architectural one).
2. A future move if scale genuinely demands multi-node scheduling or HPA.

The topology diagram below is unchanged — it already depicts the Fargate
deployment.

```
                    ┌──────────────── Route 53 + ACM (TLS) ────────────────┐
                    ▼
            ALB (+ Cognito OIDC authn)
                    │   injects signed x-amzn-oidc-data
                    ▼
      ECS Fargate service  ·  rag-app  ·  N tasks
         │        │                │
         │        │                └── Bedrock (Claude)          [LLM]
         │        └─────────────────── ElastiCache Redis          [rate limit,
         │                                                         sessions, cache]
         └──────────────────────────── Qdrant Cloud               [dense vectors]

      init / task start: pull s3://…/index/v{N}/chunks.json       [sparse source]

      Ingestion (separate ECS task, scheduled or S3-triggered):
          S3 raw docs → ingest.py → Qdrant upsert + S3 index/v{N+1}/
                                  → trigger service redeploy
```

## 4. Index distribution and the version-skew hazard

This is the sharpest operational issue in the whole design, and it is a direct
consequence of the serving architecture: **dense vectors live in Qdrant
(updated immediately by ingestion), while the sparse BM25 index is rebuilt in
each task from `chunks.json`** (`Retriever.__init__`,
[src/retrieval.py:100-102](../../src/retrieval.py)).

If ingestion upserts Qdrant and publishes a new `chunks.json` while tasks are
running, you get a **split-brain index**: running tasks serve BM25 over the old
corpus while querying a Qdrant that already holds the new one. Fused results
(RRF) then mix two corpus versions, and chunk IDs present in one but not the
other produce inconsistent citations.

**Required procedure:**

1. Ingestion writes to a **versioned, immutable prefix**: `s3://…/index/v{N}/`.
2. On success it updates a pointer object (`s3://…/index/current`) **and** forces
   a rolling redeploy of the service.
3. Each task reads the pointer **once at startup** and pins that version for its
   whole lifetime.
4. Qdrant writes and the pointer flip should be as close together as possible;
   accept a brief window of skew during the rollout, or run blue/green with a
   separate Qdrant collection per index version for zero-skew cutover.

Do **not** mount `chunks.json` from a live-mutating shared filesystem (EFS) —
that reintroduces skew without the version pin.

## 5. Identity — the one hard blocker

Today ACL roles are read from client-supplied headers
(`X-User-Id` / `X-Roles` / `X-Sites`). **On a public ALB this is trivially
spoofable**: any caller can assert any role and read restricted documents. This
must be fixed before any internet-facing deployment.

Important correction to a common assumption: **an ALB does not strip arbitrary
client headers by default.** Relying on "the load balancer will remove them" is
unsafe. The fix belongs in the application:

- ALB + Cognito authenticates the user and injects **`x-amzn-oidc-data`**, a
  signed JWT.
- The app must **verify that JWT's signature** against the ALB's public key and
  derive identity *only* from its claims.
- `AccessContext.from_headers` must **ignore** `X-User-Id` / `X-Roles` /
  `X-Sites` entirely when running in this mode.

Suggested shape: an `IDENTITY_BACKEND=headers|oidc` switch, mirroring the
existing backend-switch pattern, so on-prem keeps the trusted-header behaviour
and cloud requires verified claims.

### 5.1 Multi-customer hosting is blocked in code, not by infrastructure

Moving to cloud is exactly when someone proposes "one deployment, all
customers." **That is not safe today**, and no choice of orchestrator or
identity provider changes it — the blockers are in application code:

| Blocker | Location |
|---|---|
| `qdrant_collection` is a single global setting | [config.py:34](../../src/config.py) |
| `chunks.json` is one file holding every chunk | [retrieval.py:100-102](../../src/retrieval.py) |
| `SemanticCache` is a process-global deque — **a cache hit could serve one customer's answer to another** | [cache.py:27](../../src/cache.py) |
| Session memory is keyed by `session_id` only, with no tenant dimension | [memory.py](../../src/memory.py) |

Critically, **`enable_acl` is not a tenant boundary.** It separates roles and
sites *within* one organisation; it was never designed to isolate *between*
organisations, and should not be presented as doing so.

The gap register tracks this as Tier-4 *"multi-tenancy (per-tenant index /
quota / isolation)"* — not built. Until it is, **one deployment per customer is
the only safe model**, which is also the simplest to operate and the model the
code already supports.

## 6. Do you need GPU?

Probably not, if the LLM is Bedrock. The only local model work at serving time
is embedding the query (BGE-small) and cross-encoder reranking of ~20–40
candidates. On CPU that is on the order of tens to low hundreds of
milliseconds — acceptable at low QPS, and it makes Fargate (no GPU) viable.

GPU becomes worthwhile if you (a) self-host the LLM instead of Bedrock, (b) run
heavy Docling/OCR ingestion in-cloud, or (c) need high sustained QPS. In those
cases use ECS on the **EC2 launch type** with GPU instances (or EKS with GPU
node groups, if you later move to Kubernetes per §3).

⚠️ **Per-task concurrency is capped by design.** `_embed_lock` and
`_rerank_lock` are `threading.Lock`s
([src/retrieval.py:104-105](../../src/retrieval.py)), so embedding and reranking
are serialized within a task. A task can therefore look CPU-idle while being
concurrency-saturated. **Autoscale on ALB target response time or in-flight
request count — not on CPU utilisation.**

## 7. Observability and data residency

`/metrics` is already Prometheus-format, so AMP + AMG needs no code change, and
structured JSON logs flow to CloudWatch via the awslogs driver.

⚠️ **Langfuse Cloud sends prompts and answers off-account.** For industrial or
defence customers those payloads may contain proprietary equipment data, fault
histories and procedures. Either **self-host Langfuse** in the same VPC, or
leave `LANGFUSE_*` unset (tracing is env-gated and no-ops cleanly) and rely on
CloudWatch + AMP. Confirm against your data-classification policy before
enabling the hosted option.

## 8. Code changes required, in priority order

| # | Change | Why | Effort |
|---|---|---|---|
| 1 | Verified-JWT identity (`IDENTITY_BACKEND=oidc`) | **Security blocker.** Headers are spoofable on a public ALB (§5) | Medium |
| 2 | Durable feedback store (S3 / DynamoDB) | `FeedbackStore` appends to a local JSONL ([src/feedback.py:31](../../src/feedback.py)); per-task and **lost on task restart** | Small |
| 3 | Shared semantic cache | `SemanticCache` is an in-process `deque` ([src/cache.py:27](../../src/cache.py)); hit rate degrades ~1/N across N tasks | Small |
| 4 | `bedrock` provider | Implement the existing `providers/base.py` contract; map Bedrock throttling into `retry.py` so backoff triggers correctly | Small |
| 5 | Sparse index out of process | Removes `chunks.json` from the serving path → genuinely stateless tasks, no §4 skew, no per-task corpus RAM | Large |

Items 2–4 are small and mutually independent. Item 1 gates any public exposure.
Item 5 is discussed as the strategic option in §10.

## 9. Cost shape (directional — verify current pricing)

- **Bedrock** is per-token with no idle cost. At low/bursty volume it is far
  cheaper than keeping a GPU warm. A continuously busy deployment eventually
  crosses over to favour self-hosted inference on GPU EC2.
- **Fargate** bills per vCPU/GB-second with no control-plane fee. Avoiding EKS's
  per-cluster control-plane charge (on top of node cost) was part of *why*
  Fargate was chosen in §3, given this system's scale.
- **Qdrant Cloud** vs. self-hosted StatefulSet: managed costs more per month but
  removes HA, snapshot and upgrade work that is otherwise unbudgeted engineering
  time.
- **ElastiCache** has a floor cost. If Redis exists *only* to serve this app's
  three small needs, DynamoDB with TTL is a cheaper alternative — at the price of
  writing new backends (see §10).

## 10. Caveats, and the better long-term approach

**Caveats in this design as written:**

1. **It inherits a stateful serving model.** Every task loads the full corpus
   into RAM and rebuilds BM25 at startup. Cold start is proportional to corpus
   size, which makes autoscaling sluggish and §4's version skew unavoidable.
2. **Startup cost is real.** Models + corpus parse + BM25 build plausibly take
   30–120s. Bake the HF cache into the image (`PREFETCH_MODELS=true`, already
   supported) and set generous readiness `initialDelay` / `failureThreshold`
   against the existing `/ready`.
3. **Self-hosted Qdrant on EBS is AZ-bound.** EBS volumes are zonal; losing that
   AZ means restoring from snapshot. `backup_index.py` / `restore_index.py` give
   you the handles, but recovery is manual. Managed Qdrant or cross-AZ replicas
   remove this.
4. **Bedrock is not a drop-in for the `anthropic` provider.** It needs a
   Bedrock-specific client, AWS credentials/region, different model identifiers,
   and per-account model-access enablement. Budget for it as a small but real
   piece of work, not a config change.

**The better long-term approach — move sparse retrieval into Qdrant.**

Qdrant supports sparse vectors natively. Serving both dense *and* sparse from
Qdrant would delete `chunks.json` from the serving path entirely, and with it:

- per-task corpus RAM and the ~100k-chunk ceiling,
- the slow, corpus-proportional cold start,
- the S3 version-pinning and rolling-restart procedure of §4,
- the split-brain hazard, because both halves of the index then live in one
  store and update transactionally together.

Tasks become genuinely stateless, which is what makes Fargate/App Runner and
responsive autoscaling work properly. This is listed as "Large" in §8 because of
the re-ingest and retrieval-code change, but **for cloud specifically it is the
highest-leverage item in this document** — it removes a class of deployment
problems rather than working around them. If an AWS deployment is intended to
be long-lived, do this *before* building the §4 machinery, not after.

## 11. Open questions to resolve before building

1. **Will this AWS deployment ever serve more than one customer?** If so, the
   Tier-4 multi-tenancy work in §5.1 is a **prerequisite, not an enhancement**.
2. What is the data-classification ruling on sending prompts/answers to a hosted
   tracing provider? (§7)
3. Expected QPS and corpus size at 12 months? (Decides GPU, §6, and whether §10
   is required rather than optional.)
4. Is Cognito the intended IdP, or an existing corporate OIDC/SAML provider? (§5)
5. Is Bedrock model access already enabled in the target account and region? (§2)

> *Resolved:* on-prem Kubernetes is **not** planned — on-prem is Docker Compose,
> for both single-site multi-user and per-customer redeployment. See §3.

---

Companion documents: [cloud.md](cloud.md) (generic Kubernetes),
[on-prem-airgapped.md](on-prem-airgapped.md), [operations.md](operations.md).
