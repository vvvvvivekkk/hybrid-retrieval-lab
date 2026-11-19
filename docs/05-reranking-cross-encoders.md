# 5. Reranking — cross-encoders

**File in this repo:** `src/rerank.py`

## Bi-encoder vs cross-encoder

```
BI-ENCODER (fast, used for search)
    embed(query)      ─┐
                       ├──→ cosine → score
    embed(document)   ─┘
    Encoded separately. They never meet inside the model.

CROSS-ENCODER (accurate, used for reranking)
    ┌─────────────────────────────────────┐
    │  [CLS] query [SEP] document [SEP]   │ → transformer → score
    └─────────────────────────────────────┘
    One input. Attention runs across both texts.
```

The cross-encoder can compare *every query word against every document word* inside
the model. The bi-encoder compresses each text into a fixed vector before they ever
meet, so any detail not captured in those 384 numbers is gone.

### What that buys you

A bi-encoder genuinely struggles to tell apart:

> query: *"does vitamin D **reduce** fracture risk"*
> doc A: *"vitamin D supplementation **reduced** fracture incidence"*
> doc B: *"vitamin D supplementation **increased** fracture incidence"*

Both documents are about vitamin D and fractures. Their embeddings sit almost on top
of each other. A cross-encoder reads the query and the document together and can
actually weigh "reduce" against "increased."

### What it costs

Nothing can be precomputed. With a bi-encoder you embed 5,183 documents once at
startup; every query is then one cheap matrix multiply. With a cross-encoder, scoring
query Q against 5,183 documents means **5,183 full forward passes through a
transformer, per query.** It's not slightly slower — it's thousands of times slower.

## The architecture that makes it affordable

```
5,183 documents
      ↓  BM25 + dense + fusion       ~33 ms
   top 25 candidates
      ↓  cross-encoder               ~1,700 ms  (CPU; far less on GPU)
   final ranking
```

Cheap retrievers cut the field by 99.5%. The expensive model reads only what survived.
This **cascade** pattern — cheap-and-broad, then expensive-and-narrow — shows up all
over ML systems engineering, not just retrieval.

## The measured result: there is no default reranker

This is the most interesting finding in the project. Both rerankers, both datasets,
80 queries, top-25 reranked:

| reranker | SciFact | NFCorpus |
|---|---|---|
| none (baseline) | 0.7721 | 0.4108 |
| `ms-marco-MiniLM-L-6-v2` | 0.7468 **(−0.0253)** | 0.4197 **(+0.0089)** |
| `bge-reranker-base` | 0.7969 **(+0.0248)** | 0.3627 **(−0.0482)** |

**Each model helps on one dataset and hurts on the other.** Read that table again —
it's a perfect inversion.

### Why

A cross-encoder is trained on a specific *distribution* of (query, document, relevance)
triples, and it learns what "relevant" looks like **in that distribution**.

- **MiniLM** is trained on MS MARCO: short, conversational web-search queries against
  web passages. NFCorpus queries are literally that — real queries typed by real
  people on a nutrition website. So it helps there. SciFact's formal scientific claims
  are nothing like MS MARCO, so it hurts.
- **bge-reranker-base** trains on broader data including scientific text. Opposite
  profile, opposite result.

### The traps this kills

**"Use the bigger model."** `bge-reranker-base` has ~5× the parameters and is the
*worst* option on NFCorpus — and 6× slower than the model that beats it there
(20.9 s/query vs 3.7 s/query on CPU).

**"Reranking improves RAG."** It's a bet on distribution match. On SciFact with
MiniLM it cost 1.7 s/query to make results measurably worse.

**"My gain is real."** MiniLM's NFCorpus win is **+0.0089** — under 1%, on 80 queries,
with no significance testing. That's plausibly noise, and it costs 3.7 s/query. An
honest engineer reports that caveat rather than the headline.

## Recall is the ceiling

A reranker only **reorders what you hand it**. It cannot retrieve.

If Recall@25 is 0.80 — meaning 20% of correct documents never made it into the
candidate set — then **80% is your hard maximum**, no matter how good the reranker is.

This is the answer to the most common complaint in applied RAG: *"I added a reranker
and nothing improved."* Usually it isn't a reranker problem at all. The right
documents never reached the candidate list. **Check Recall@k at stage one before you
touch the reranker.**

## The code

```python
pairs = [(query, corpus[d]["full"]) for d in doc_ids]
scores = self.model.predict(pairs, batch_size=self.batch_size)
ranked = sorted(zip(doc_ids, map(float, scores)), key=lambda x: -x[1])
```

Note `max_length=512` in the constructor. Transformers have a token limit; anything
past it is **silently truncated**. If your documents are long, the cross-encoder may
be scoring only their first half — another invisible failure worth knowing about.

And in `compare_rerankers.py`:

```python
run[q] = head + [d for d, _ in fused[q] if d not in set(head)]
```

The reranker only touches the top 25. Everything below keeps its fusion order and is
appended back. Without this, Recall@100 would be computed over a 25-document list and
would look artificially broken. **Small bookkeeping details like this are where
evaluation code quietly goes wrong.**

→ Next: [Metrics: measuring retrieval](06-metrics-how-to-measure.md)
