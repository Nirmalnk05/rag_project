# AWS Deployment — Design Proposal

> **Status: design proposal, not a verified runbook.** Nothing here has been
> executed against a live AWS account. Service names, IAM shapes and pricing
> must be re-checked against current AWS docs before you build.
>
> **Claims about this codebase are cited to file and line** so the next reader
> can re-verify rather than trust. Several service choices in §2 need
> application changes first; those are §9. The review log records what has been
> checked and when.
>
> The generic Kubernetes guide in [cloud.md](cloud.md) remains the source of
> truth for manifest-level detail; this document is the AWS-specific binding
> and the reasoning behind it.

## 1. Guiding decision: one architecture, swapped backing services

We deploy **the same container and the same application architecture** on-prem
and in AWS, and swap only the *backing services* behind the existing
environment switches. We do **not** fork the design.

| Seam | Env var | On-prem | AWS |
|---|---|---|---|
| LLM backend | `LLM_PROVIDER` | `ollama` | `bedrock` (**new — §9 item 8**) or `openai_compatible` via proxy |
| Session memory | `MEMORY_BACKEND` | `memory` | `redis` |
| Rate limiting | `RATE_LIMIT_BACKEND` | `memory` | `redis` |
| Parsing | `PARSE_BACKEND` | `hybrid` (the default) | **`local` — set it explicitly, see §6.3** |
| Guardrails | `GUARDRAILS_BACKEND` | `heuristic` | `heuristic` |

⚠️ **The parsing row is not what it looks like.** The default is
`parse_backend="hybrid"` ([config.py:103](../../src/config.py)), not `local`.
In hybrid mode the dispatcher parses locally and, when the extracted text falls
below `cloud_parse_min_chars` (200), **re-sends the whole document to LlamaParse
or Azure Document Intelligence**
([parsing/dispatcher.py:87-92](../../src/parsing/dispatcher.py)). It escalates
only when credentials are present, so today's safety rests on an unset API key
rather than on a decision anyone made. Make it a decision — §6.3.

### 1.1 Seams this design assumes that do not exist yet

| Assumed binding | Reality in the code | Resolution |
|---|---|---|
| Identity provider | No `IDENTITY_BACKEND`. Identity is derived from client headers only ([api.py:170-174](../../src/api.py)) | §9 item 1 |
| Vector-store endpoint | Only `QDRANT_HOST` / `QDRANT_PORT` ([config.py:32-33](../../src/config.py)) — no URL, API key or TLS, so **managed Qdrant cannot be addressed at all** | §9 item 2 |
| Index location | `CHUNK_STORE_PATH` is a hardcoded relative `Path("data/chunks.json")` in both [retrieval.py:21](../../src/retrieval.py) and [ingestion.py:24](../../src/ingestion.py) | **No code change needed** — §4.3 |

**What must stay identical in both environments:** retrieval logic, chunking,
prompts (`PROMPT_VERSION`) and the eval suite. If those drift, a quality
regression found in one environment will not reproduce in the other, and the
RAGAS/CI gates stop being meaningful. This constraint has teeth: it is why
§11's sparse-retrieval change cannot be made for cloud alone.

**What is allowed to differ:** compute platform, replica counts, storage,
secrets delivery, identity provider, LLM backend.

## 2. Service mapping

| Concern | AWS service | Code prerequisite | Notes |
|---|---|---|---|
| Compute | **ECS Fargate** | — | Decided — §3. On-prem is Docker Compose, so EKS's parity argument does not apply |
| Vector store | **Qdrant Cloud (AWS region)** | **§9 item 2** | Unaddressable today: all three clients use host/port with no API key. Also see the residency consequence in §6.2 |
| Shared state | **ElastiCache for Redis** | — | Rate-limit counter and session memory. The semantic cache joins it only after §9 item 9 — see §5.3 |
| LLM | **Amazon Bedrock** (Anthropic models) | **§9 item 8** | Requires explicit model-access enablement per account/region |
| Embeddings / rerank | In-process `sentence-transformers` | — | BGE-small + bge-reranker-base. CPU is adequate at low QPS — §7.1 |
| Index artifacts | **S3** | Entrypoint (§4.3); backups §9 item 7 | Versioned prefix holding all seven artifacts — §4 |
| Ingestion | **ECS task** (scheduled or S3 → EventBridge triggered) | — | `scripts/ingest.py` unchanged. Not Lambda — Docling/OCR dependency size and runtime exceed its limits |
| Identity | **ALB + Cognito (OIDC)** | **§9 items 1, 3** | Replaces today's spoofable header ACL — §5. Browser and machine callers share paths — §5.2 |
| Secrets | **Secrets Manager** / SSM Parameter Store | — | Injected as env vars; never baked into the image |
| Metrics / logs | **AMP** + **AMG**, CloudWatch Logs | — | `/metrics` is already Prometheus-format, and unauthenticated — deny it at the ALB and scrape inside the VPC (§5.2) |
| Tracing | Self-hosted Langfuse *or* disabled | — | §8 — data residency |

**§2 is not buildable until §9 item 2 lands**: the current Qdrant client cannot
address a managed endpoint. The index-artifact row needs an ops artifact rather
than a code change (§4.3).

## 3. Compute: ECS Fargate (decided)

**On-prem runs Docker Compose, and no Kubernetes rollout is planned** — for
either multiple users at a single customer site or redeployment to separate
customers. That settles this choice.

EKS's one strong argument here was reusing `deploy/k8s/` for genuine
on-prem/cloud parity. With no on-prem Kubernetes, that argument disappears
entirely, leaving only its costs: a per-cluster control-plane charge, cluster
upgrades, and add-on/IRSA/CNI operations.

The decision rests on **single-digit replica counts and a small operations
team**. It deliberately does *not* rest on the corpus ceiling: that ceiling is a
property of the current serving design which §11 proposes to remove, so it
should not be load-bearing in a compute choice.

`deploy/k8s/` stays in the repo as a maintained *optional* path, for two cases:

1. A customer mandates deployment into their own existing Kubernetes cluster
   (a procurement constraint, not an architectural one).
2. A future move if scale genuinely demands multi-node scheduling or HPA.

```
                    ┌──────────────── Route 53 + ACM (TLS) ────────────────┐
                    ▼
        ALB  ·  app.example.com → authenticate-oidc   (browser + UI)
             ·  api.example.com → no OIDC, API key    (machines, §5.2)
                    │   injects signed x-amzn-oidc-data
                    ▼
      ECS Fargate service  ·  rag-app  ·  N tasks  (private subnets)
         │        │                │
         │        │                └── Bedrock (Claude)      [VPC endpoint]
         │        └─────────────────── ElastiCache Redis      [rate limit, sessions]
         └──────────────────────────── Qdrant                 [dense + chunk payloads]
                                        └─ Cloud ⇒ NAT egress, off-account (§6.2)

      entrypoint: resolve pointer → s3 sync index/v{N}/ → /app/data/  (§4.3)

      Ingestion (separate ECS task, scheduled or S3-triggered):
          S3 raw docs → ingest.py → Qdrant collection v{N+1} + S3 index/v{N+1}/
                                  → atomic pointer flip → rolling redeploy
```

### 3.1 Task sizing

A derivable floor, in order of size:

- The full `chunks.json` corpus held as Python dicts **plus** the `BM25Okapi`
  term structures built over every chunk
  ([retrieval.py:100-102](../../src/retrieval.py)). At the documented
  ~100k-chunk ceiling this dominates and is plausibly several GB.
- The embedder and cross-encoder resident in torch together with the runtime
  (`BAAI/bge-small-en-v1.5` + `BAAI/bge-reranker-base`,
  [config.py:37-38](../../src/config.py)) — order of 0.5–1 GB.
- FastAPI/uvicorn and the per-request working set.

**Start at 2 vCPU / 8 GB, then measure.** The numbers above are a shape, not a
measurement, and this is measurable today rather than at deploy time: run the
app against the production corpus and read the process RSS once `/ready` returns
200 (`docker stats`, or `ps -o rss= -p <pid>` inside the container). Set memory
to roughly 1.5× observed peak and re-measure whenever the corpus grows
materially. Expect 16 GB at the upper end of the ceiling; 2 vCPU on Fargate
supports up to 16 GB, beyond which the vCPU count has to rise too.

Fargate ephemeral storage defaults to 20 GiB and holds the image plus the HF
cache, which matters for §11 caveat 2.

## 4. Index distribution and the version-skew hazard

This is the sharpest operational issue in the design, and a direct consequence
of the serving architecture: **dense vectors live in Qdrant (updated
immediately by ingestion), while the sparse BM25 index is rebuilt in each task
from `chunks.json`** (`Retriever.__init__`,
[retrieval.py:100-102](../../src/retrieval.py)).

If ingestion upserts Qdrant and publishes a new `chunks.json` while tasks are
running, you get a **split-brain index**: tasks serve BM25 over the old corpus
while querying a Qdrant that already holds the new one. Fused results (RRF)
then mix two corpus versions, and chunk IDs present in one but not the other
produce inconsistent citations.

### 4.1 Choosing a cutover strategy

Pinning each task to one index version does **not** remove the skew — it makes
each task internally consistent while leaving it inconsistent with Qdrant. A
task pinned to `v{N}` serves BM25 over `v{N}` for its *whole lifetime* against a
Qdrant already at `v{N+1}`. During a rolling deploy that lasts the full drain of
every old task, and if the deployment stalls or is rolled back, indefinitely.

| Option | Skew | Cost |
|---|---|---|
| Shared mutable `chunks.json` on EFS | Unbounded and invisible | Lowest |
| Versioned S3 prefix + per-task pin + rolling restart | Bounded by drain time, but **present on every old task for the whole rollout** | Low |
| **Blue/green: a new Qdrant collection per index version** | **None** | A full republish per index version — see below |

**Recommendation: blue/green**, as the only option that eliminates the hazard.

1. Ingestion writes index `v{N+1}` to a **new Qdrant collection** *and* a new
   immutable S3 prefix `s3://…/index/v{N+1}/`.
2. On success it flips **one** pointer object naming *both* halves — collection
   name and S3 prefix.
3. Each task reads that pointer once at startup and takes both halves from the
   same version, so a task can never hold one version of one half (§4.3).
4. A rolling redeploy then cuts over; old tasks stay fully self-consistent on
   `v{N}` until they drain. Drop the old collection after a retention window.

Note that during the rollout two self-consistent corpus versions are serving at
once, so a user's successive questions can be answered from different versions.
Session history is shared through Redis rather than pinned to a task, so a
follow-up question can arrive at a task whose corpus no longer matches the
citations in that history. This is far milder than mixed-version fusion, but if
a customer tracks answer stability, say so in the release notes and keep
rollouts short.

**Blue/green is not free — it fights incremental ingestion.** Ingestion is
designed to be incremental: `ingest_manifest.json` plus deterministic uuid5
point IDs mean a re-run upserts only what changed, into one collection. A new
collection per version means every vector is written again on every publish, so
near the ~100k-chunk ceiling each publish costs a full re-upload — and a full
re-embed unless vectors are carried across. Two ways out, both needing a
decision:

- **Snapshot and delta** — snapshot the live collection, restore it under the
  new version's name, then apply only the manifest delta. Cheaper per publish,
  more moving parts, and it needs a tested restore path.
- **Accept the full republish** — simplest, and fine when ingestion is
  infrequent (nightly or weekly) and the corpus sits near the low end of §3.1.

If neither is acceptable, fall back to the pinned-prefix row and accept bounded
skew during rollouts — but record that as a decision rather than discovering it
in production.

### 4.2 Version the prefix, not the file

There are **seven** index artifacts
([backup_index.py:17-18](../../scripts/backup_index.py)), and all of them must
move together:

| Artifact | Why it must share the version |
|---|---|
| `chunks.json` | BM25 source and chunk text |
| `ingest_manifest.json` | Incremental-ingest state; a stale manifest re-ingests or skips the wrong documents |
| `context_cache.json` | Paid contextual-retrieval LLM output — lose it and re-ingestion re-pays |
| `caption_cache.json` | Paid vision output — same |
| `index_meta.json` | Index provenance, including which embed model produced the vectors |
| `acl.json` | The ingest-time ACL sidecar. Mismatched with its chunks it means **wrong visibility**, not just wrong ranking |
| `versions.json` | Document version / supersede state |

`acl.json` is what makes this a security property, not only a quality one.

Do **not** mount these from a live-mutating shared filesystem (EFS) — that
reintroduces skew without the version pin.

### 4.3 This needs a container entrypoint, not application code

Worth stating plainly, because it is easy to over-scope: **§4 requires no change
to the application.**

- `Retriever()` is constructed with no arguments in the lifespan handler
  ([api.py:39](../../src/api.py)), so it reads the default relative path
  `data/chunks.json`.
- `.dockerignore` already excludes `data/chunks.json` from the image — the
  container is built expecting an index to be supplied at run time.
- The image sets `WORKDIR /app` and has a bare `CMD` with no `ENTRYPOINT`.

So a startup script placed before `uvicorn` is sufficient:

1. read the pointer object,
2. `aws s3 sync s3://…/index/v{N}/ /app/data/`,
3. `export QDRANT_COLLECTION=<collection for v{N}>`,
4. `exec uvicorn src.api:app --host 0.0.0.0 --port 8000`.

`Settings` has no `env_prefix`
([config.py:11](../../src/config.py)), so `QDRANT_COLLECTION` maps to
`qdrant_collection`; `get_settings()` is `lru_cache`d and `dense_search` reads
`s.qdrant_collection` per call
([retrieval.py:146](../../src/retrieval.py)), so exporting the variable before
the process starts pins the collection for the task's lifetime — exactly the
guarantee §4.1 step 3 needs. `/app/data` already exists and is owned by the
non-root `raguser`, so the fetch can write there
([Dockerfile:21-22](../../Dockerfile)).

**It is not literally free: the image has no S3 client.** `requirements.txt`
contains no `boto3`, and the `python:3.12-slim` base installs no AWS CLI, so the
entrypoint needs one added — `boto3` (a few MB, Python entrypoint) rather than
`awscli` (~100 MB, and §11 caveat 2 is about image size on Fargate). An
entrypoint script also has to be copied in and made executable. So this is a
**build and ops change, not an application change** — which is why it stays off
§9's list, and why it still needs an owner.

⚠️ **Fetch atomically.** A partially completed sync leaves the task with
artifacts from two versions — the §4.2 hazard, reintroduced at startup. Sync to
a temporary directory, verify all seven artifacts are present, then move into
place; fail the task rather than starting degraded. Give the task role read
access to that one prefix.

## 5. Identity and access

### 5.1 The header ACL is spoofable, and the fix has three parts

Today ACL roles are read from client-supplied headers
(`X-User-Id` / `X-Roles` / `X-Sites`). **On a public ALB this is trivially
spoofable**: any caller can assert any role and read restricted documents. This
must be fixed before any internet-facing deployment.

A correction to a common assumption: **an ALB does not strip arbitrary client
headers by default.** "The load balancer will remove them" is unsafe. The fix
belongs in the application, and it has three parts — all three, or it buys
nothing:

1. **Verify the JWT.** ALB + Cognito authenticates the user and injects
   `x-amzn-oidc-data`, a signed JWT. The app must verify its signature against
   the ALB's regional public key and derive identity *only* from its claims;
   `AccessContext.from_headers` must **ignore** `X-User-Id` / `X-Roles` /
   `X-Sites` entirely in this mode.
2. **Turn ACLs on.** `_access_for()` returns `None` whenever `enable_acl` is
   false ([api.py:170-174](../../src/api.py)), and the default *is* false
   ([config.py:70](../../src/config.py)). OIDC with ACLs off authenticates the
   user and then shows them everything.
3. **Map claims to the dimensions the retriever filters on.** `acl_visible()`
   reads `acl_roles` / `acl_sites` from chunk metadata and treats a chunk with
   neither as **public** ([access.py:39-50](../../src/access.py)). So you need
   Cognito groups → roles/sites *and* a populated `data/acl.json` at ingest
   ([ingestion.py:40](../../src/ingestion.py)). Without this part every chunk
   stays public and parts 1–2 change nothing observable.

Suggested shape: an `IDENTITY_BACKEND=headers|oidc` switch mirroring the
existing backend-switch pattern, so on-prem keeps trusted-header behaviour and
cloud requires verified claims.

### 5.2 Three credentials, and the browser is the hard case

The chat UI is served by **this same application** at `/`
([api.py:112](../../src/api.py)), and every endpoint that does real work already
requires an API key — `/ask`, `/ask/stream`, `/feedback`, `/session/reset`
([api.py:128,144,156,177,219](../../src/api.py)). The UI obtains that key from a
text box and keeps it in `localStorage`
([static/index.html:52,65-66](../../src/static/index.html)).

Adding ALB + Cognito therefore puts three credentials in play on the same paths,
and the design has to say which authorizes what:

| Credential | Issued by | What it should establish |
|---|---|---|
| Cognito session cookie | ALB | That a human is signed in |
| `x-amzn-oidc-data` JWT | ALB | **Identity for ACL filtering** (§5.1) |
| `X-API-Key` | Us, shared | Machine callers only |

**Browser traffic must stop needing `X-API-Key`.** Otherwise SSO users still
paste a shared secret into a text box — the exact thing OIDC was meant to
remove — and that secret sits in `localStorage` on every user's machine. So
`require_api_key` has to accept a verified JWT as an alternative, and the API-key
box comes out of the UI. That is **§9 item 3**, and it lands with item 1 or the
rollout is worse than what it replaces.

**Machine callers cannot be separated by path**, because they use the same ones
the UI does. Separate them by **hostname** — a second listener or ALB,
`api.example.com` with no `authenticate-oidc` and the API key enforced in-app —
or by a dedicated prefix the UI never calls. A rule over `/ask` alone will
either lock out the eval harness and CI, or leave the UI unauthenticated.

- ALB **health checks bypass listener rules** — they hit the target directly —
  so `/ready` works either way.
- `/metrics` carries no auth dependency at all
  ([api.py:117](../../src/api.py)). **Deny it at the ALB** and scrape it from
  inside the VPC.

⚠️ **SSE versus the ALB idle timeout.** `/ask/stream`
([api.py:218](../../src/api.py)) holds a long-lived connection, and the agent
loop can legitimately run `agent_max_steps` (6) × `llm_timeout_s` (60s)
([config.py:60](../../src/config.py), [config.py:27](../../src/config.py)) — up
to roughly six minutes. The ALB's default idle timeout is 60 seconds. Raise it
above your worst-case generation time, or streams get cut mid-answer.

### 5.3 Multi-customer hosting is blocked in code, not by infrastructure

Moving to cloud is exactly when someone proposes "one deployment, all
customers." **That is not safe today**, and no choice of orchestrator or
identity provider changes it — the blockers are in application code:

| Blocker | Location |
|---|---|
| `qdrant_collection` is a single global setting | [config.py:34](../../src/config.py) |
| `chunks.json` is one file holding every chunk | [retrieval.py:100-102](../../src/retrieval.py) |
| Session memory is keyed by `session_id` only, with no tenant dimension | [memory.py:66](../../src/memory.py) |
| `SemanticCache` is a process-global deque, and is disabled whenever ACLs are on | [cache.py:27](../../src/cache.py), [orchestrator.py:43-45](../../src/orchestrator.py) |

On that last row: cached answers are **never** served across principals, because
`_cache_active()` is `self._cache.enabled and access is None`
([orchestrator.py:43-45](../../src/orchestrator.py)) and `_access_for()` returns
`None` only when `enable_acl` is false
([api.py:170-174](../../src/api.py)). The consequence is not a leak — it is that
**enabling ACLs, which §5.1 requires for cloud, disables the semantic cache
entirely.** Sharing the cache across tasks is therefore wasted work until the
cache is made ACL-aware (§9 item 9), and ElastiCache should not be sized or
justified for it before then.

Critically, **`enable_acl` is not a tenant boundary.** It separates roles and
sites *within* one organisation; it was never designed to isolate *between*
organisations, and should not be presented as doing so.

Per-tenant index, quota and isolation are **not built, and not tracked anywhere
in this repository** — treat them as unscoped work, not as a backlog item
someone else is holding. Until they exist, **one deployment per customer is the
only safe model**, which is also the simplest to operate and the model the code
already supports.

### 5.4 Revoking access requires a re-ingest

ACLs come from an ingest-time sidecar
([ingestion.py:40](../../src/ingestion.py)) baked into chunk metadata, which
`acl_visible()` then reads off the chunk. So revocation is not a config change:
edit `data/acl.json` → re-ingest the affected documents → publish a new index
version → roll the service (§4). **Expect access changes to take a deploy, not
a minute.** If a customer's security requirements include prompt revocation,
that is a design change (query-time ACL lookup), not an operational one — and
it is not in §9.

## 6. Networking and data egress

### 6.1 VPC shape

Tasks in **private subnets with no public IP**; ALB in public subnets; Qdrant
(if self-hosted) and ElastiCache in private subnets, with security groups that
admit only the app's security group.

Add VPC endpoints so this traffic never touches the internet and does not incur
NAT processing: **S3** (gateway), and interface endpoints for **Bedrock
runtime**, **Secrets Manager**/SSM, **ECR** (api *and* dkr) and **CloudWatch
Logs**.

Put a **WAF** on the public ALB. Neither the endpoints nor the WAF exist as IaC
anywhere in this repo — see §11 caveat 5.

### 6.2 What still leaves the VPC: the corpus

**Qdrant Cloud sits outside your VPC and holds the full chunk text and
vectors** — that is, the corpus. Reaching it means NAT egress to a third
party's account. A deployment can be entirely private-subnet and still ship
every maintenance document's content to another company. **That is the single
largest residency consequence of §2, and it is a decision, not a networking
detail.**

If the residency ruling is no: self-host Qdrant in private subnets and accept
the HA/backup/upgrade work (§11 caveat 3).

⚠️ **Self-hosted Qdrant has no authentication in this codebase.** All three
client constructions are plaintext HTTP with no API key —
[retrieval.py:118](../../src/retrieval.py),
[ingestion.py:238](../../src/ingestion.py),
[backup_index.py:35](../../scripts/backup_index.py) — and config exposes only
host and port. Anything that can reach port 6333 can read or delete the whole
collection. At minimum constrain it with a security group; properly, add
API-key + TLS support (§9 item 2), which is the same change that unlocks
managed Qdrant.

### 6.3 Parsing egress — make it an explicit decision

Per §1, `PARSE_BACKEND=hybrid` escalates low-confidence documents to LlamaParse
or Azure Document Intelligence, logged at info level, with no per-document
record of what left the account.

For AWS: set **`PARSE_BACKEND=local`** and leave `LLAMAPARSE_API_KEY` /
`AZURE_DI_*` unset, unless the data-classification ruling explicitly permits
third-party parsing. If it does, prefer Azure DI in your own tenant over
LlamaParse, and budget for egress auditing. Relying on an absent API key is not
a control — someone will set it to fix a parsing bug.

### 6.4 Egress summary, ranked by sensitivity

| Path | What leaves the account | Default |
|---|---|---|
| **Qdrant Cloud** | Full corpus — chunk text and vectors | On, if §2 is followed |
| **Cloud parsing** | Whole source documents | **On** in `hybrid` whenever credentials exist |
| Bedrock | Prompts and retrieved context | On — in-AWS, same region |
| Langfuse Cloud | Prompts and answers | Off unless `LANGFUSE_*` is set |

Whether in-AWS counts as leaving the account at all is a posture question, not a
technical one — §12 question 2 decides it, and the answer may differ for a
defence customer.

## 7. Scale, concurrency and failure behaviour

### 7.1 Do you need GPU?

Probably not, if the LLM is Bedrock. The only local model work at serving time
is embedding the query (BGE-small) and cross-encoder reranking of ~20–40
candidates. On CPU that is on the order of tens to low hundreds of
milliseconds — acceptable at low QPS, and it makes Fargate (no GPU) viable.

GPU becomes worthwhile if you (a) self-host the LLM instead of Bedrock, (b) run
heavy Docling/OCR ingestion in-cloud, or (c) need high sustained QPS. In those
cases use ECS on the **EC2 launch type** with GPU instances (or EKS with GPU
node groups, if you later move to Kubernetes per §3).

### 7.2 Per-task concurrency is capped by design

`_embed_lock` and `_rerank_lock` are `threading.Lock`s
([retrieval.py:104-105](../../src/retrieval.py)), so embedding and reranking are
serialized within a task. A task can therefore look CPU-idle while being
concurrency-saturated. **Do not autoscale on CPU utilisation.**

### 7.3 Autoscaling: set a floor, don't chase the curve

Two things to get right:

- ECS Application Auto Scaling's predefined ALB metric is
  `ALBRequestCountPerTarget`. **There is no predefined target-tracking metric
  for response time.** Scaling on latency needs a step-scaling or custom
  target-tracking policy against the ALB's `TargetResponseTime` CloudWatch
  metric. Don't put it in a ticket as if it were a checkbox.
- At a 30–120s cold start (§11 caveat 2) a reactive policy adds capacity *after*
  the burst has passed.

**At single-digit replica counts, over-provision a fixed desired-count for
peak** and use scaling only to raise the floor slowly (and to scale in
overnight). Responsive autoscaling is worth tuning only once §11's
sparse-out-of-process change makes tasks start fast.

### 7.4 Failure behaviour: readiness over-reports, and one dependency fails everything

Two code-level gaps that matter more in AWS than on-prem, because Qdrant moves
from the next container to across a NAT gateway:

**`/ready` does not check its dependencies.** It returns 200 as soon as the
retriever and orchestrator objects exist
([api.py:102-109](../../src/api.py)) — there is no Qdrant or Redis probe. A task
reports healthy to the ALB while Qdrant is unreachable, and the ALB sends it
traffic. **§9 item 5.**

**A Qdrant failure fails the whole query.** `hybrid_retrieve` runs sparse and
dense under `asyncio.gather` with no `return_exceptions=True`
([retrieval.py:213-216](../../src/retrieval.py)), so a dense-side exception
propagates and the request fails — even though the sparse half is local, needs
nothing over the network, and would still return citable results. Degrading to
BM25-only turns a third-party outage into visibly weaker answers instead of
downtime. **§9 item 4.**

Both are small, independent of whether AWS happens, and worth doing on-prem too.

## 8. Observability and data residency

`/metrics` is already Prometheus-format, so AMP + AMG needs no code change
(scrape inside the VPC, not through the ALB — §5.2), and structured JSON logs
flow to CloudWatch via the awslogs driver.

⚠️ **Langfuse Cloud sends prompts and answers off-account.** For industrial or
defence customers those payloads may contain proprietary equipment data, fault
histories and procedures. Either **self-host Langfuse** in the same VPC, or
leave `LANGFUSE_*` unset (tracing is env-gated and no-ops cleanly) and rely on
CloudWatch + AMP.

Decide this alongside the other three egress paths in §6.4 rather than on its
own — it is the smallest of them.

## 9. Code changes required, in priority order

| # | Change | Why | Effort |
|---|---|---|---|
| 1 | Verified-JWT identity, ACL enablement, and claims→ACL mapping (`IDENTITY_BACKEND=oidc`) | **Security blocker.** All three parts of §5.1; any one alone is ineffective | Medium |
| 2 | Qdrant URL + API key + TLS (`QDRANT_URL`, `QDRANT_API_KEY`) | **Prerequisite for §2's vector store.** Managed Qdrant is unaddressable today, and self-hosted runs unauthenticated (§6.2) | Small |
| 3 | Accept a verified JWT in place of `X-API-Key` for browser traffic, and remove the API-key box from the chat UI | Lands with item 1, or SSO users still paste a shared secret that lives in `localStorage` (§5.2) | Small |
| 4 | Retrieval degradation: `return_exceptions=True` plus a BM25-only fallback | A Qdrant hiccup currently fails the whole query, though the sparse half needs no network (§7.4) | Small |
| 5 | `/ready` probes Qdrant and Redis | Tasks currently report healthy while a critical dependency is unreachable (§7.4) | Small |
| 6 | Durable feedback store (S3 / DynamoDB) | `FeedbackStore` appends to a local JSONL ([feedback.py:31](../../src/feedback.py)); per-task and **lost on task restart** | Small |
| 7 | `backup_index.py` writes to S3 | It `shutil.copy2`s into a local directory today ([backup_index.py:27-31](../../scripts/backup_index.py)); §2 claims S3 | Small |
| 8 | `bedrock` provider | Implement the `providers/base.py` contract, and map Bedrock throttling to `LLMRetryableError` — `call_with_retries` backs off only on that type and re-raises `LLMError` terminally ([retry.py:28-34](../../src/providers/retry.py)) | Small |
| 9 | **ACL-aware** semantic cache, *then* share it in Redis | Sharing alone is a no-op: §5.1's ACLs disable the cache (§5.3). Key by principal/role-set first | Medium |
| 10 | Sparse index out of process | Removes `chunks.json` from the serving path → genuinely stateless tasks, no §4 skew, no per-task corpus RAM | Large |

Items 1–3 gate any public deployment and ship together. Items 4–5 are small,
raise availability, and apply on-prem as well. Items 6–8 are independent of each
other. Item 9 must precede any work on a shared cache. Item 10 is the strategic
option in §11.

The index fetch is deliberately absent: it needs a container entrypoint, not
application code (§4.3).

## 10. Cost shape (directional — verify current pricing)

- **Bedrock** is per-token with no idle cost. At low/bursty volume it is far
  cheaper than keeping a GPU warm. A continuously busy deployment eventually
  crosses over to favour self-hosted inference on GPU EC2.
- **Fargate** bills per vCPU/GB-second with no control-plane fee. Avoiding EKS's
  per-cluster charge was part of *why* Fargate was chosen (§3). Note that §7.3's
  over-provisioning advice means paying for peak capacity most of the time.
- **Qdrant Cloud** vs. self-hosted: managed costs more per month but removes HA,
  snapshot and upgrade work that is otherwise unbudgeted engineering time.
- **Ingestion republishes** are a real line item under §4.1's blue/green: each
  index version writes the whole collection again. Price it per publish against
  your ingest cadence before choosing between blue/green and snapshot-and-delta.
- **NAT and cross-account egress** is easy to forget: every Qdrant Cloud query
  and every cloud-parse upload is NAT-processed data. The VPC endpoints in §6.1
  remove the S3/Bedrock/Secrets share; the Qdrant Cloud share you cannot remove
  without self-hosting.
- **ElastiCache** has a floor cost. It serves rate limiting and session memory
  only until §9 item 9 lands; if that is all it ever does, DynamoDB with TTL is
  cheaper, at the price of writing new backends.

## 11. Caveats, and the better long-term approach

**Caveats in this design as written:**

1. **It inherits a stateful serving model.** Every task loads the full corpus
   into RAM and rebuilds BM25 at startup. Cold start is proportional to corpus
   size, which makes autoscaling sluggish (§7.3) and §4's skew structural.
   ([cloud.md](cloud.md) opens by calling the API "stateless and scales
   horizontally." That is true of request handling and false of the process —
   each replica holds the whole sparse index.)
2. **Startup cost is real — and baking models into the image is the wrong lever
   on Fargate.** `PREFETCH_MODELS=true` ([Dockerfile:12](../../Dockerfile))
   genuinely removes the model download. But Fargate has no host-level image
   cache: every task start pulls the image over the network, so adding torch
   plus two models turns a one-off per-host download into a multi-GB **pull on
   every scale-out**. On ECS/EC2 the baked image is a clear win. On Fargate,
   either pair it with a SOCI index for lazy pulling, or keep the image lean and
   absorb a first-request model download behind generous `/ready` thresholds.
   Measure the pull before assuming baking helps.
3. **Self-hosted Qdrant on EBS is AZ-bound.** EBS volumes are zonal; losing that
   AZ means restoring from snapshot. `backup_index.py` / `restore_index.py` give
   you the handles, but recovery is manual. Managed Qdrant or cross-AZ replicas
   remove this.
4. **Bedrock is not a drop-in for the `anthropic` provider.** It needs a
   Bedrock-specific client, AWS credentials/region, different model identifiers,
   and per-account model-access enablement. Budget it as a small but real piece
   of work, not a config change.
5. **This document is architecture, not a deployable plan.** §6.1 names the
   subnet shape, endpoints and WAF you need; it does not specify security-group
   rules or IAM policies, and there is **no Terraform/CDK anywhere in this
   repo**. Someone has to own that before a build date means anything.

**The better long-term approach — move sparse retrieval into Qdrant.**

Qdrant supports sparse vectors natively. Serving both dense *and* sparse from
Qdrant would delete `chunks.json` from the serving path entirely, and with it:

- per-task corpus RAM and the ~100k-chunk ceiling,
- the slow, corpus-proportional cold start,
- the S3 version-pinning and blue/green machinery of §4,
- the split-brain hazard, because both halves of the index then live in one
  store and update transactionally together.

Tasks become genuinely stateless, which is what makes Fargate and responsive
autoscaling work properly.

⚠️ **But this cannot be a cloud-only change.** §1 requires retrieval logic to be
identical in both environments, precisely so a quality regression reproduces in
both and the RAGAS gates stay meaningful. Moving sparse retrieval into Qdrant
changes ranking, so doing it for AWS alone would make the two deployments answer
differently and quietly void the shared golden set. Scope it as a change to
**both** environments, including:

- an on-prem re-ingest, since both must produce sparse vectors at ingest time,
- a scoring review: Qdrant's sparse scoring is not `BM25Okapi`'s formula. IDF
  weighting is available, but the k1/b length saturation that `rank_bm25`
  applies is not the same function, so retrieved sets will shift.
- **re-baselining the RAGAS thresholds**, because the scores will move.

Scoped honestly it is still the highest-leverage item in this document — it
removes a class of deployment problem rather than working around it. If an AWS
deployment is intended to be long-lived, do this *before* building §4's
blue/green machinery, not after.

## 12. Open questions to resolve before building

1. **Will this AWS deployment ever serve more than one customer?** If so, the
   per-tenant isolation work in §5.3 is a **prerequisite, not an enhancement**.
2. **What is the data-classification ruling on each of §6.4's four egress
   paths?** Not just tracing: Qdrant Cloud takes the corpus, `hybrid` parsing
   takes whole documents, and whether in-AWS Bedrock counts as egress at all is
   part of the same ruling.
3. Expected QPS and corpus size at 12 months? (Decides GPU §7.1, task sizing
   §3.1, and whether §11's sparse migration is required rather than optional.)
4. Is Cognito the intended IdP, or an existing corporate OIDC/SAML provider? And
   which machine callers need the separate hostname in §5.2?
5. Is Bedrock model access already enabled in the target account and region?
6. Who owns the IaC and the IAM policy set, and does any exist for on-prem
   today? (§11 caveat 5)
7. Must access revocation be immediate? If yes, §5.4 makes it a design change.
8. What is the ingest cadence? It decides blue/green versus snapshot-and-delta
   in §4.1 and the republish cost in §10.

> *Resolved:* on-prem Kubernetes is **not** planned — on-prem is Docker Compose,
> for both single-site multi-user and per-customer redeployment. See §3.

## Review log

Claims in this document have been checked against the codebase on the dates
below. Corrections are recorded here rather than in the body, so the body reads
as the current design.

**2026-10-30, first pass.** Corrected: the `PARSE_BACKEND` seam, which defaults
to `hybrid` and escalates documents to a third-party parser, and its residency
consequence (§1, §6.3, §6.4); the assumption that managed Qdrant was reachable,
and the absence of any Qdrant authentication (§2, §6.2); a claim that the
semantic cache could leak across customers — it cannot, and the real consequence
is that ACLs disable it (§5.3); a claim that version pinning bounded skew to a
brief window — it does not (§4.1); versioning of `chunks.json` alone, which
omitted `acl.json` and two paid caches (§4.2); attribution of multi-tenancy
tracking to a non-existent "gap register" (§5.3); a claim that `backup_index.py`
targeted S3, which writes locally (§9 item 7); the autoscaling metric advice
(§7.3); and the framing of the sparse migration as cloud-specific, which
contradicted §1 (§11). Added: §1.1 missing seams, §3.1 task sizing, §5.2
machine traffic and the SSE/idle-timeout hazard, §5.4 revocation latency, §6
networking and egress, and §11 caveat 5.

**2026-10-30, second pass.** Corrected: the index fetch, previously listed as a
code change gating deployment — it needs only a container entrypoint, since
`Retriever()` takes the default path and `.dockerignore` already omits the index
(§4.3); the two-listener-rule plan in §5.2, which assumed the UI and machine
callers could be split by path when they share paths and every endpoint already
requires an API key (§5.2, §9 item 3); and the claim that Qdrant sparse search
lacks the corpus statistics `rank_bm25` provides — IDF weighting is available,
and the real risk is a different scoring formula (§11). Added: the republish cost
of blue/green and its conflict with incremental ingestion (§4.1, §10, §12 q8);
§7.4 on readiness over-reporting and the absence of retrieval degradation (§9
items 4–5); the unauthenticated `/metrics` endpoint (§2, §5.2); a measurement
procedure for task sizing (§3.1); and neutral wording on whether in-AWS traffic
counts as egress (§6.4). Removed the draft-to-draft commentary that the first
pass left in the body.

---

Companion documents: [cloud.md](cloud.md) (generic Kubernetes),
[on-prem-airgapped.md](on-prem-airgapped.md), [operations.md](operations.md).

Diagrams: [aws-hld.drawio](../design/aws-hld.drawio) (deployment topology) and
[aws-lld.drawio](../design/aws-lld.drawio) (request path · index publish · task
startup). They mark anything from §9 as not built, so they read as a proposal
rather than as current state. The on-prem equivalents are
[hld.drawio](../design/hld.drawio) and [lld.drawio](../design/lld.drawio).
