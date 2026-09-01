# AI Exception Engine

Learns what normal operational behaviour looks like **in context**, and creates a case
only when something meaningfully deviates from it.

Not fraud detection, not a SIEM, not threshold alerts. The distinguishing idea is
contextual: a 2,000 EUR refund is unremarkable for an electronics seller and
extraordinary for a bookseller, so a single global threshold produces either noise or
silence.

---

## The rule that shapes the architecture

> The AI layer never sets a score and never decides that something is an exception.

Every anomaly score is produced by a deterministic ensemble of detectors and is
**reproducible from the stored detector outputs** — the exception detail screen
recomputes its own score in front of you. The model in `app/ai` is called only after
detection, receives only computed evidence, and returns prose. With no provider
configured the product is fully functional.

## What it does

1. **Ingests** events, validated against a schema the tenant registers.
2. **Fits contextual baselines** for every combination of context dimensions, with
   fallback to broader scopes when a scope is too thin to judge.
3. **Scores** each event with six independent detectors.
4. **Raises exceptions** only when the score, the confidence, an explanation, the
   policies and the deduplication all agree it deserves a human.
5. **Explains** every case: what was observed, what was expected, and which population
   it was compared against.
6. **Finds similar past cases** and what was done about them.
7. **Learns from feedback** — into a proposal somebody has to promote.

## Quick start

```bash
docker compose up
```

- API and interactive docs: <http://localhost:8000/docs>
- UI: <http://localhost:3000>

The demo workspace seeds itself: 4,000 synthetic e-commerce events with six planted
anomalies and two deliberate traps.

### Without Docker

```bash
cd backend && pip install -r requirements-dev.txt && uvicorn app.main:app --reload
```

```bash
cd frontend && npm install && npm run dev
```

The backend defaults to SQLite, so no database server is needed locally.

## Tests

```bash
cd backend && python -m pytest -q
```

126 tests cover the baselines, all six detectors, the ensemble, exception generation,
the full HTTP surface, the feedback loop and tenant isolation.

The ones that matter most assert behaviour on the demo data: each planted anomaly is
found, **and each planted trap is not**.

## The six detectors

| Detector | Notices | Why it earns its place |
|---|---|---|
| **Robust deviation** | A value far from its scope's median | Median/MAD, not mean/σ: a few large refunds would otherwise inflate the very spread used to detect them. |
| **Categorical rarity** | Values and combinations nobody has seen here | Smoothed by sample size — unseen in 40 observations is far weaker evidence than unseen in 40,000. |
| **Isolation forest** | Points odd only in combination | Fitted per scope, so "odd in combination" still means odd *for this context*. |
| **Rolling quantile** | Departure from the entity's own recent behaviour | Catches drift a static baseline misses. |
| **Change point** | A sustained shift, not a spike | Binary segmentation on a mean-shift cost; stays quiet for single outliers, which the deviation detectors already cover. |
| **Peer group** | An entity unlike comparable entities | Some abnormalities exist only in aggregate: a seller refunding 55% of orders looks ordinary order by order. |

## Design decisions worth knowing

Most of these exist because the first implementation got them wrong and the demo data
showed it.

**Zero-inflated features are handled explicitly.** A refund amount is `0` on every order
that was not refunded. Left alone those structural zeros dominate: the median is 0, the
spread collapses, and *every ordinary refund* scores as a wild outlier. Statistics are
computed over the non-zero population, and a zero is read as "the thing did not happen"
rather than as an extreme low value. Before this fix, 18 of 18 exceptions were this
artefact.

**The ensemble is not a vote.** Detectors are independent tests, so a weighted mean is
the wrong model: with six detectors, five of which have nothing to say about a
particular problem, one certain detector gets diluted into silence — a seller refunding
60% against a peer median of 12% scored **29 out of 100**. The strongest detector sets
the floor; agreement from the others lifts it.

**Abstaining is not scoring zero.** A detector without enough history says so, and is
excluded from the calculation entirely. Otherwise thin data quietly drags every score
down — precisely when new entities look suspicious.

**Comparisons never leave the context.** When an entity has too little history in its
own scope, the comparison falls back through coarser scopes and then stops. It never
compares a luxury order against the same seller's books: that is not a weaker
comparison, it is a wrong one, and it manufactures exactly the false positives the
product exists to avoid.

**A fallback peer group only compares scale-free metrics.** A luxury seller's average
refund is legitimately ten times a bookseller's; the *share* of orders refunded stays
comparable.

**An exception nobody can explain is never raised.** A detector reports a score even
when no single feature crossed its reporting threshold, and agreement between near-
misses can push the total over the bar. The result is a case with a number and no
reason — worse than no case, because nobody can action it or check it.

**Deduplication distinguishes entity-level from event-level findings.** "This seller
refunds more than its peers" does not become new information because another order
arrived, so while it is open a repeat is always folded in. The fingerprint is keyed on
the *leading* detector: keying it on every reason made it unstable, and one seller
produced eleven identical exceptions.

**One entity cannot own the queue.** The per-entity cap counts everything still open,
not just recent arrivals — windowing it lets an entity wait out the window and file the
next one.

**A suppression policy carries a score ceiling.** Accepting "these are fine" must never
become a blind spot for a worse version of the same shape.

**Feedback never changes production behaviour on its own.** Labels accumulate into a
proposal with measured per-detector precision; somebody promotes it, or does not. One
analyst clicking "false positive" must not be able to silently blind the system.

**Models are fitted per scope, not per event.** Refitting an isolation forest for every
event is O(n) model fits over a batch — it turned an 8-second run into minutes of pure
retraining for an identical answer.

### Deviations from the original brief

- **No `ruptures`.** Change-point detection is ~50 lines of binary segmentation on a
  mean-shift cost, deterministic and directly testable, which removes a dependency.
- **No Polars.** The core works over plain event dictionaries; a dataframe library
  would add weight without adding capability at this size. scikit-learn is used, for the
  isolation forest only.
- **A sixth detector was added.** The brief lists peer-group anomalies as an exception
  type but only five detectors; none of the five can see an aggregate property of an
  entity, so the planted peer-group anomaly went undetected until `peer_group` existed.

## Repository layout

```
backend/app/
  core/           config, database session, tenant resolution, structured logging
  baselines/      contextual baselines, robust statistics, scope fallback
  detectors/      the detector interface, the five per-event detectors, peer groups
  scoring/        the ensemble and the batch pipeline
  exceptions/     generation gates, deduplication, service layer
  similarity/     structured similarity between exceptions
  feedback/       labels, measured precision, retraining proposals
  policies/       suppression (models live in app/models.py)
  notifications/  pluggable delivery with a recording default
  ai/             provider abstraction + structured narrative generation
  demo/           seeded dataset with six planted anomalies and two traps
frontend/
  app/            Next.js App Router pages (server components)
```

## The demo data

Six planted anomalies — point, contextual, peer-group, change, frequency burst, rare
combination — and two deliberate traps:

- **Luxury orders** are genuinely large: extreme globally, normal in scope. A detector
  that flags them is not contextual, it is a threshold with extra steps.
- **A new seller** with eight events. Being new is not the same as being abnormal.

On 4,000 events the engine raises **10 exceptions**, all six detectors contribute, both
traps stay silent, and every score recomputes exactly.

## Multi-tenancy

Every table carries `tenant_id`, every query filters on it, and the tenant is resolved
from an `X-API-Key` header before any handler runs. Cross-tenant access returns 404, not
403. Tests assert it for reads, ingestion and detection runs.

## What is not built

No streaming ingestion (batch only). No background job queue — detection runs
synchronously, fine to a few hundred thousand events. No embedding-based similarity for
free-text notes (the hook exists; structured similarity covers the current fields). No
RBAC beyond a per-tenant key. Notification delivery is an abstraction with a recording
default rather than real Slack/email integrations.
