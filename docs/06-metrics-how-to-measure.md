# 6. Metrics — how to measure retrieval

**File in this repo:** `src/metrics.py`

Without metrics you are guessing. "The results look better" is not engineering — it's
confirmation bias with extra steps. You looked at five queries you picked yourself.

## What you need: an answer key

Measurement requires **qrels** ("query relevance judgements") — a human-labelled map
saying which documents are correct for which query.

```
query_id  →  {doc_id: relevance_score}
"q7"      →  {"doc_31715818": 1, "doc_4983": 1}
```

Relevance can be binary (0/1) or graded (0 = irrelevant, 1 = related, 2 = perfect).
SciFact is mostly binary; NFCorpus is graded.

This is why the project uses **BEIR** — a standard collection of retrieval benchmarks
that ships with qrels. Building your own means a human reading documents for days.
For your own production system you will eventually have to do exactly that, on a few
hundred real queries. It is worth it.

## nDCG@10 — the headline metric

Normalised Discounted Cumulative Gain. Build it one word at a time:

**Gain** — each relevant document in your results earns points equal to its relevance.

**Cumulative** — add them up.

**Discounted** — points shrink with position, since a correct answer at rank 1 is
worth more to a user than the same answer at rank 9. The discount is `1/log₂(rank+1)`:

| rank | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| weight | 1.000 | 0.631 | 0.500 | 0.387 | 0.289 |

Logarithmic, not linear — rank 1 vs 2 is a big drop, rank 9 vs 10 barely matters.
That matches how people actually read a results page.

**Normalised** — divide by the best score achievable for *that* query (its **IDCG**,
ideal DCG), so queries with 1 correct answer and queries with 30 are comparable, and
every score lands in [0, 1].

### Worked example

Top 5 results. Correct documents at positions **2** and **4**. Binary relevance,
2 relevant documents total.

```
DCG  = 1/log₂(2+1) + 1/log₂(4+1)
     = 1/1.585     + 1/2.322
     = 0.631       + 0.431       = 1.062

IDCG (both at top) = 1/log₂(2) + 1/log₂(3)
                   = 1.000     + 0.631       = 1.631

nDCG@10 = 1.062 / 1.631 = 0.651
```

Same two documents at positions 1 and 2 → nDCG = 1.0. Same documents at 9 and 10 →
about 0.20. **The metric cares where things are, not just whether they're present.**

```python
def ndcg_at_k(ranked_ids, relevant, k=10):
    dcg = sum(relevant.get(d, 0) / math.log2(i + 1)
              for i, d in enumerate(ranked_ids[:k], start=1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
```

`relevant.get(d, 0)` returns 0 for documents not in the answer key — correct, since an
unjudged document contributes no gain.

## Recall@k — the ceiling metric

Of all correct documents, what fraction appeared **anywhere** in your top k?

```python
hits = len(set(ranked_ids[:k]) & set(relevant))
return hits / len(relevant)
```

Position-blind by design. It answers one question: *did we even find them?*

**This is the most operationally important metric in a RAG pipeline**, for the reason
in [doc 5](05-reranking-cross-encoders.md): a reranker can only reorder what it
receives. Recall@50 = 0.9517 means **95.17% is the hard ceiling** on everything
downstream. No reranker, no prompt, no bigger model recovers the missing 4.83%.

**Diagnostic rule:** if Recall@k is high but nDCG is low, the documents are there and
badly ordered → fix ranking, add a reranker. If Recall@k is low, ranking is irrelevant
→ fix retrieval first.

## MRR@10 — the "one answer" metric

Mean Reciprocal Rank: `1 / (position of the first correct answer)`, averaged.

| first correct at rank | 1 | 2 | 3 | 4 | 10 | not in top 10 |
|---|---|---|---|---|---|---|
| contribution | 1.00 | 0.50 | 0.33 | 0.25 | 0.10 | 0 |

Use it when only the top result matters — a chatbot quoting one source, a "jump to
answer" feature. Ignores everything after the first hit, which is the point.

## Choosing

| situation | metric |
|---|---|
| General ranking quality, comparing to published work | **nDCG@10** |
| First-stage retrieval feeding a reranker or LLM | **Recall@k** |
| Only the single top result is shown | **MRR@10** |

Report more than one. A system that wins on nDCG and loses on Recall is telling you
something real — exactly what happened with fusion weights in
[doc 4](04-fusion-combining-retrievers.md).

## The aggregation detail

```python
out[f"nDCG@{k}"] = sum(ndcg_at_k(run[q], qrels[q], k) for q in run) / len(run)
```

Per-query score, then averaged — **macro-averaging**. Every query counts equally,
whether it has 1 correct document or 40.

The alternative (pooling all hits then computing once, **micro-averaging**) lets
queries with many relevant documents dominate the number. Macro is standard in
retrieval and it's what BEIR leaderboards use, so this repo's numbers are comparable
to published ones. Mixing them up is a common way to produce a score nobody else can
reproduce.

→ Next: [Evaluation and leakage](07-evaluation-and-leakage.md)
