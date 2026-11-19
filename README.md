# hybrid-retrieval-lab

**A hybrid search system — keyword + semantic + fusion + reranking — built from scratch, measured on two datasets, with leakage-free tuning.**

Most RAG tutorials assemble these components and declare victory. This repo measures
what each one actually buys you, and the answer is uncomfortable: **three of the four
"standard" upgrades did nothing or made things worse, and which reranker to use
flips depending on the corpus.**

---

## What this project is about

When you ask an AI about your own documents, the model doesn't read all of them.
Something finds the relevant passages first and hands them over. That something is a
**retriever**, and it quietly decides whether the final answer is any good — if it
hands over the wrong passages, no model can save the answer.

### The four pieces

**1. BM25 — the keyword searcher.** Scores a document by how often your query's words
appear, weighting rare words more heavily ("the" tells you nothing, "hyperkalemia"
tells you a lot) and penalising long documents so they don't win just by being big.
It has no idea what words *mean* — to BM25, "car" and "automobile" are unrelated.

**2. Dense vector search — the meaning searcher.** An *embedding model* turns each
passage into 384 numbers, arranged so similar meanings land close together. Your
question becomes numbers too; nearest documents win. Matches paraphrases with zero
shared words. Its weakness mirrors BM25's: it blurs exact tokens, so rare identifiers
get smoothed into "something vaguely related", confidently.

**3. Reciprocal Rank Fusion — merging the two.** BM25 returns scores like `18.4`,
dense returns `0.81`. Different scales, so you can't add them. RRF throws the scores
away and uses only *positions*: `score = Σ 1/(k + rank)`. Documents both retrievers
rank highly float to the top. `k` controls how sharply early ranks dominate.

**4. Cross-encoder reranking — the accuracy pass.** The dense retriever embeds query
and document *separately*, so it can't notice your question asks about a *decrease*
while the passage reports an *increase*. A cross-encoder reads both together. Far more
accurate, far slower — so it only sees the top ~25 candidates.

### Why measurement is the actual project

Wiring these together is an afternoon. Knowing whether each one *helped* needs an
answer key. This repo uses two BEIR benchmarks with deliberately different character:

| | docs | queries | what a query looks like |
|---|---|---|---|
| **SciFact** | 5,183 | 300 | a formal scientific claim; relevant = a paper supports/refutes it |
| **NFCorpus** | 3,633 | 323 | a real search query typed by a real person on a nutrition site |

Metrics are implemented from scratch in `src/metrics.py`: **nDCG@10** (correct docs in
the top 10, worth more the higher they sit, normalised 0–1), **Recall@k**, **MRR@10**.

---

## Results

**All tuning is done on train/validation and reported on test.** See Finding 1.

### SciFact
| stage | nDCG@10 | Recall@50 | p50 latency |
|---|---|---|---|
| BM25 only | 0.6653 | 0.8692 | 12 ms |
| Dense only | 0.7077 | 0.9250 | 21 ms |
| **Hybrid (tuned on train: k=10, w=0.5:1)** | 0.7069 | **0.9517** | 33 ms |
| *hybrid, best-on-test (leaked)* | *0.7221* | — | — |

### NFCorpus
| stage | nDCG@10 | Recall@50 |
|---|---|---|
| BM25 only | 0.3078 | 0.2073 |
| Dense only | 0.3408 | 0.2571 |
| **Hybrid (tuned on validation: k=10, w=0.5:1)** | **0.3562** | **0.2642** |

**Hybrid's benefit is not universal.** On NFCorpus it beats dense outright (+4.5%
nDCG). On SciFact it does *not* improve nDCG at all — it ties — and earns its place
only through recall (+2.7 points), which matters solely if something downstream
consumes the extra candidates.

The same tuned config (`k=10`, weights `0.5:1`) won on both datasets independently,
so that much does transfer.

---

## Finding 1 — tuning on the test set inflated the headline by 2.2%

The first version of this repo swept `k` across the **test** set and reported the best
value found: nDCG@10 = 0.7188. That is leakage — the number describes how well a config
fitted that particular answer key, not how the system handles unseen queries.

`tune.py` implements the honest protocol: grid-search on train/validation → freeze the
single winner → run it once on test.

| | SciFact | NFCorpus |
|---|---|---|
| best-on-test (leaked) | 0.7221 | 0.3562 |
| tuned on train/val (honest) | **0.7069** | **0.3562** |
| inflation | **+2.16%** | +0.00% |

2.2% is roughly the margin that separates adjacent entries on a public leaderboard.
Note NFCorpus shows **zero** inflation — validation and test picked the same config.
**Leakage isn't a fixed tax; it's a variance effect**, large on small/narrow eval sets
and invisible on well-behaved ones. You cannot tell which you have without checking.

## Finding 2 — the RRF constant everyone copies is a hyperparameter

Every blog post uses `k=60` from the original 2009 paper. Almost nobody tunes it.

| rrf_k | nDCG@10 (SciFact) |
|---|---|
| 60 | 0.7040 |
| 10 | **0.7188** |

Large `k` flattens the scoring curve so rank 1 and rank 20 score almost the same,
letting a weaker retriever drag documents down from the top of a stronger one's list.
Small `k` protects precision at the top. Tuned honestly, `k=10` won on both datasets.

## Finding 3 — there is no default reranker; it inverts by corpus

80 queries, top-25 reranked, tuned fusion config on both.

| reranker | SciFact | NFCorpus |
|---|---|---|
| none (baseline) | 0.7721 | 0.4108 |
| `ms-marco-MiniLM-L-6-v2` | 0.7468 **(−0.0253)** | 0.4197 **(+0.0089)** |
| `bge-reranker-base` | 0.7969 **(+0.0248)** | 0.3627 **(−0.0482)** |

**Each model helps on one dataset and hurts on the other.** The bigger, "better"
reranker is not better — `bge-reranker-base` is the worst option on NFCorpus by a
wide margin, and 6× slower than the model that beats it there.

Why: MiniLM is trained on MS MARCO — short conversational web-search queries.
NFCorpus queries are exactly that; SciFact's formal claims are nothing like it.
`bge-reranker-base` trains on broader data including scientific text, which is why
the ranking flips.

**The takeaway isn't "pick bge" or "pick MiniLM". It's that a reranker is a
distribution bet, and the only way to know is to run both on your own data.** Note
also that MiniLM's win on NFCorpus is +0.0089 for 3.7 s/query on CPU — a gain so
small it may not survive its own cost.

## Finding 4 — Recall@k is the ceiling on everything downstream

A reranker can only reorder what it is handed. SciFact Recall@50 is 0.9517, so **95.2%
is the hard ceiling** regardless of reranker quality. Optimise recall in stage one;
stage-one nDCG matters less because the reranker resorts the top anyway. This is also
the answer to the most common complaint in RAG work — *"my reranker isn't helping."*
Usually it isn't a reranker problem. The right documents never reached the candidate set.

## Finding 5 — post-filtering metadata silently destroys recall

Two ways to apply a filter like "only 2024 docs" or "only docs this user may see":
**post-filter** (retrieve top-k, then drop non-matching) or **pre-filter** (restrict
candidates before scoring). Measured with synthetic partitions, requesting top-20:

| filter keeps | post-filter: results returned | pre-filter: results returned |
|---|---|---|
| 100% | 20.0 | 20.0 |
| 50% | 9.9 | 20.0 |
| 20% | 3.9 | 20.0 |
| 5% | **0.9** | 20.0 |
| 1% | **0.2** | 20.0 |

Post-filtering is one line of code, which is why teams ship it. At a 5% filter it
returns **0.9 results instead of 20** — and only users with narrow filters ever see
the bug, so it survives testing.

## Finding 6 — an ANN index is not free speed, and not needed at this scale

HNSW is a graph index that walks toward the query instead of scanning everything.
The honest way to report one is recall-against-exact *and* latency, never latency alone:

| index | recall@100 vs exact | ms/query |
|---|---|---|
| flat (exact) | 1.000 | 0.339 |
| hnsw ef=16 | 0.691 | 0.082 |
| hnsw ef=128 | 0.990 | 0.183 |
| hnsw ef=256 | 0.999 | 0.304 |

The 4.6× speedup only exists at 69% recall, which is useless. At usable recall it's
1.9×, and both numbers are a fraction of a millisecond — irrelevant beside the 21 ms
the query embedding already costs. **ANN earns its place at millions of documents,
not thousands.** Shipping it here would be cargo-culting.

---

## Architecture

```
query
  ├─> BM25           (keyword — exact terms)     ──┐
  └─> dense vectors  (meaning — paraphrases)     ──┤
                                                   ├─> RRF fusion (k=10, w 0.5:1) ─> top 25
                                                   ┘                                   │
                                      optional cross-encoder rerank (corpus-dependent) <┘
```

| file | what it does |
|---|---|
| `src/load.py` | BEIR loading, split-aware (train / validation / test) |
| `src/sparse.py` | BM25 with stopword filtering |
| `src/dense.py` | bge-small embeddings, cosine search |
| `src/fusion.py` | weighted Reciprocal Rank Fusion |
| `src/rerank.py` | cross-encoder reranking |
| `src/ann.py` | FAISS flat + HNSW indexes, recall-vs-exact measurement |
| `src/metadata.py` | pre-filter and post-filter implementations |
| `src/metrics.py` | nDCG@k, Recall@k, MRR@k from scratch |
| `tune.py` | **leakage-free tuning** — grid on train, report on test |
| `run_ablation.py` | stage-by-stage ablation |
| `compare_rerankers.py` | reranker head-to-head |
| `benchmark_ann.py` | ANN recall/latency curve |
| `benchmark_metadata.py` | pre- vs post-filter recall collapse |

Every module is commented to explain *why*, not just *what*.

## Running it

```bash
pip install -r requirements.txt
python download_data.py --dataset scifact
python download_data.py --dataset nfcorpus

python cache_runs.py  --dataset scifact --device cuda   # index once, cache runs
python tune.py        --dataset scifact                 # honest tuning
python compare_rerankers.py --dataset scifact --device cuda
python benchmark_ann.py      --dataset scifact --device cuda
python benchmark_metadata.py --dataset scifact --device cuda
```

Any BEIR dataset works — `--dataset fiqa`, `trec-covid`, `arguana`. On an RTX 4060
the full suite is a few minutes; on CPU the cross-encoders alone take ~20 s/query.

## Remaining limitations

- **Two datasets, both small and both scientific-ish.** FiQA (financial, 57k docs)
  is the obvious third and runs with `--dataset fiqa`; it was skipped here only
  because this machine had no GPU.
- **Reranker comparisons use 80 queries**, not the full test set — enough to show a
  sign flip, not enough for a tight confidence interval.
- **Metadata partitions are synthetic.** The recall-collapse mechanism is real and
  dataset-independent, but these specific numbers are not a claim about real metadata.
- **No statistical significance testing.** Differences under ~0.01 nDCG (e.g. MiniLM's
  +0.0089 on NFCorpus) should be treated as noise until bootstrapped.
- **Single embedding model.** `bge-small-en-v1.5` throughout; swapping it would likely
  move more than any fusion tuning did.

## License

MIT
