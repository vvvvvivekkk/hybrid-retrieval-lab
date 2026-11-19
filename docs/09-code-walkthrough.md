# 9. Code walkthrough

Every file, what it does, and why it's shaped that way.

## Layout

```
hybrid-retrieval-lab/
├── src/                      reusable library code
│   ├── load.py               BEIR data loading, split-aware
│   ├── sparse.py             BM25
│   ├── dense.py              embeddings
│   ├── fusion.py             RRF
│   ├── rerank.py             cross-encoders
│   ├── metrics.py            nDCG / Recall / MRR
│   ├── ann.py                FAISS flat + HNSW
│   └── metadata.py           pre/post filtering
├── download_data.py          fetch a BEIR dataset
├── cache_runs.py             index once, cache ranked lists
├── tune.py                   leakage-free hyperparameter search
├── run_ablation.py           stage-by-stage comparison
├── sweep_weights.py          diagnostic trade-off sweep
├── compare_rerankers.py      reranker head-to-head
├── benchmark_ann.py          index recall/latency curve
├── benchmark_metadata.py     filtering comparison
└── data/<dataset>/           corpus, qrels, caches, results
```

**The split matters.** `src/` is library code — importable, no side effects, no
`print`. Top-level scripts are experiments — they have `argparse`, print tables, and
write JSON. Mixing the two is how projects become unrunnable six months later.

---

## `src/load.py`

Loads the three BEIR pieces: **corpus**, **queries**, **qrels** (the answer key).

```python
out[str(r["_id"])] = {"title": title, "text": text,
                      "full": (title + ". " + text).strip(". ").strip()}
```

`full` glues title to body. Titles carry disproportionate retrieval signal — they're
dense with topic words — so both retrievers search over the combined text.

```python
def load_qrels(dataset="scifact", split="test"):
    if s > 0:
        out.setdefault(...)[...] = s
```

Scores of 0 mean "judged and **not** relevant" — different from unjudged. Dropping
them keeps the structure to "what is correct," which is what the metrics expect.

The `split` parameter is the whole reason this file was rewritten — see
[doc 7](07-evaluation-and-leakage.md).

## `src/sparse.py`

BM25. Covered in detail in [doc 2](02-bm25-keyword-search.md).

The one piece of engineering worth repeating:

```python
idx = np.argpartition(-scores, k - 1)[:k]
idx = idx[(-scores[idx]).argsort()]
```

Partial sort — O(n) to isolate the top k, then sort only those. Fully sorting 5,183
scores to read 100 is wasted work, and the habit matters more as corpora grow.

## `src/dense.py`

Embeddings. Three things to notice:

```python
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
```
Required by the BGE family. Omitting it costs accuracy silently.

```python
normalize_embeddings=True
```
Makes cosine similarity equal the dot product, so search is one matrix multiply.

```python
self.emb = self.model.encode([...], show_progress_bar=True)
```
The entire corpus is embedded **once** at construction. This is the expensive step
(~9 minutes on CPU, seconds on a GPU) and the reason `cache_runs.py` exists.

## `src/fusion.py`

Twelve lines. Covered in [doc 4](04-fusion-combining-retrievers.md).

```python
for rank, (doc_id, _score) in enumerate(results, start=1):
    fused[doc_id] += w / (k + rank)
```

`_score` is deliberately discarded — using only rank is the whole point. The
underscore prefix signals "intentionally unused," which stops a future reader thinking
it's a bug.

## `src/metrics.py`

nDCG, Recall, MRR, implemented from scratch — see [doc 6](06-metrics-how-to-measure.md).

Written by hand rather than imported from `pytrec_eval` on purpose: you should be able
to derive your own evaluation metric. If you can't explain what nDCG does, you can't
tell when it's misleading you.

## `src/rerank.py` · `src/ann.py` · `src/metadata.py`

Covered in [doc 5](05-reranking-cross-encoders.md) and
[doc 8](08-scaling-indexes-and-filters.md).

---

## The scripts

### `download_data.py`
```bash
python download_data.py --dataset scifact
```
Fetches corpus, queries, and **every** qrels split. Prints which splits exist, since
some BEIR datasets have train/validation/test and others only test.

### `cache_runs.py` — the one that makes iteration tolerable

Builds both indexes, runs every query through both retrievers, and saves the ranked
lists to JSON.

**Why this exists:** embedding the corpus is minutes of work. Fusion tuning, weight
sweeps and reranker comparisons only need the *ranked lists* — not the models. Pay
once, then every downstream experiment is instant.

```python
json.dump({"splits": wanted, "runs": cache}, ...)
```
Stores which query IDs belong to which split, so `tune.py` can honour them without
re-reading qrels.

```python
ap.add_argument("--max-train", type=int, default=250)
```
Hyperparameter search needs far fewer queries than final evaluation. Capping train
keeps caching fast without weakening the test numbers.

### `tune.py` — the most important script here

Grid → train → freeze → test. See [doc 7](07-evaluation-and-leakage.md). It also
prints the leaked number alongside, so the inflation is measurable rather than
theoretical.

### `run_ablation.py`

Runs BM25 → dense → hybrid → hybrid+rerank on the same queries, reporting accuracy
**and** p50/p95 latency.

```python
head = [d for d, _ in reranked]
runs["hybrid+rerank"][qid] = head + [d for d, _ in fused if d not in set(head)]
```

The reranker touches only the top N; the tail keeps fusion order and is appended back.
Without this, Recall@100 would be computed over a 25-item list. **Evaluation
bookkeeping is where silent errors live.**

```python
L = sorted(lat[name]); p50, p95 = L[len(L)//2], L[int(len(L)*0.95)]
```

p50 and p95, not the mean. Latency distributions have long tails and the mean hides
them — users feel p95.

### `sweep_weights.py`

Diagnostic only. Shows the shape of the `k` / weight trade-off across a split. It is
**not** how configs are chosen — `tune.py` is. Keeping exploration and evaluation in
separate files is what stops leakage creeping back in.

### `compare_rerankers.py` · `benchmark_ann.py` · `benchmark_metadata.py`

Each isolates one question and answers it with a table. Note that
`compare_rerankers.py` reads the tuned config from `tuned.json` rather than hardcoding
it — experiments build on the honest result, not a number someone typed.

---

## The flow, end to end

```
download_data.py          corpus + queries + qrels → data/<dataset>/
        ↓
cache_runs.py             build BM25 + dense indexes (expensive, once)
                          run every query → cache ranked lists
        ↓
tune.py                   grid-search on train → freeze → score on test
                          → data/<dataset>/tuned.json
        ↓
    ┌───────────────┬──────────────────┬────────────────────┐
compare_rerankers  benchmark_ann   benchmark_metadata   run_ablation
```

Everything after `cache_runs.py` runs in seconds, because the expensive work is done.
**Structuring a project so the slow part happens once is a habit worth stealing.**

→ Next: [Running and extending](10-running-and-extending.md)
