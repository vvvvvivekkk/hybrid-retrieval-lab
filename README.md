# hybrid-retrieval-lab

**A hybrid search system — keyword + semantic + fusion + reranking — built from scratch and measured at every stage.**

The headline finding is a negative one, and that is the point: two components that
every RAG tutorial tells you to add made retrieval **worse**. This repo shows the
measurement that caught it, the diagnosis, and the fix.

---

## What this project is about

When you ask an AI a question about your own documents, the model doesn't actually
read all of them. Something has to find the handful of relevant passages first and
hand them over. That "something" is a **retriever**, and it is the part of a RAG
system that quietly decides whether the answer is any good. If retrieval hands over
the wrong passages, no model on earth can save the answer.

This project builds a retriever properly and proves how well it works with numbers.

### The four pieces

**1. BM25 — the keyword searcher.**
Scores a document by how often your query's words appear in it, weighting rare words
more heavily (the word "the" tells you nothing; "hyperkalemia" tells you a lot) and
penalising long documents so they don't win just by being big. It has no idea what
words *mean* — to BM25, "car" and "automobile" are unrelated. That blindness is
exactly why it needs a partner.

**2. Dense vector search — the meaning searcher.**
An *embedding model* turns each passage into a list of 384 numbers, arranged so that
texts with similar meaning land close together in that 384-dimensional space. Your
question becomes numbers too, and the nearest documents win. This matches paraphrases
with zero shared words — "how do I get my money back" finds a passage about refund
policy. Its weakness is the mirror image of BM25's: it blurs exact tokens, so rare
identifiers and form numbers get smoothed into "something vaguely related", and it
returns plausible-but-wrong results with total confidence.

**3. Reciprocal Rank Fusion — merging the two.**
Now you have two ranked lists. BM25 returns scores like `18.4`; the dense retriever
returns scores like `0.81`. Different scales, so you can't just add them, and
normalising is fragile. RRF solves this by throwing the scores away and using only
*positions*:

```
score(doc) = sum over each list of  1 / (k + rank_in_that_list)
```

Documents that both retrievers rank highly float to the top. `k` controls how sharply
early ranks are favoured.

**4. Cross-encoder reranking — the accuracy pass.**
The dense retriever embeds the query and the document *separately*, so the model never
sees them together and can't notice that your question asks about a *decrease* while
the passage reports an *increase*. A **cross-encoder** feeds both texts in as one
input and reads them jointly. Far more accurate — and far slower, because nothing can
be precomputed. So you only ever run it on the top ~50 candidates, not all 5,183
documents.

### Why the measurement matters more than the code

Wiring these four together is an afternoon's work. Knowing whether each one *helped*
requires an answer key: a set of queries where the correct documents are already
labelled. This project uses **BEIR/SciFact** (5,183 scientific abstracts, 300 labelled
queries), which makes every claim below checkable rather than vibes.

Three metrics are implemented from scratch in `src/metrics.py`:

- **nDCG@10** — rewards correct documents in the top 10, worth more the higher they
  sit, normalised so 0 = useless and 1 = perfect. The standard metric in retrieval papers.
- **Recall@k** — of all correct documents, what fraction appeared anywhere in the top k.
- **MRR@10** — 1/(position of the first correct answer).

---

## Results

5,183 documents, 300 labelled queries. Latency is per query, on CPU.

| stage | nDCG@10 | MRR@10 | Recall@50 | p50 latency |
|---|---|---|---|---|
| BM25 only | 0.6653 | 0.6339 | 0.8692 | 12 ms |
| Dense only (bge-small-en-v1.5) | 0.7077 | 0.6758 | 0.9250 | 21 ms |
| Hybrid, RRF k=60 *(textbook default)* | 0.7040 | 0.6725 | 0.9443 | 33 ms |
| **Hybrid, RRF k=10 *(tuned)*** | **0.7188** | **0.6880** | **0.9443** | **33 ms** |
| Hybrid + MiniLM reranker | 0.7468 † | — | — | +1,569 ms |
| Hybrid + bge-reranker-base | 0.7936 † | — | — | +9,688 ms |

† Reranker rows are measured on an 80-query subset where the **no-rerank baseline
scores 0.8101** — meaning both rerankers *lost* to simply not reranking.

**Recommended production config: hybrid with RRF k=10, no reranker.**
+1.6% nDCG and +2.1 points of Recall@50 over dense alone, for 12 ms. The reranker
costs 50–300× the latency to make results worse on this corpus.

---

## Finding 1 — the RRF constant everyone copies is a hyperparameter

Every blog post uses `k=60`, the value from the original 2009 RRF paper. Almost nobody
tunes it.

| rrf_k | weights (sparse : dense) | nDCG@10 | Recall@50 |
|---|---|---|---|
| 60 | 1 : 1 | 0.7040 | 0.9443 |
| 10 | 1 : 1 | **0.7188** | 0.9443 |
| 10 | 0.5 : 1 | 0.7069 | **0.9517** |

At k=60 the hybrid system **loses to dense retrieval alone**. At k=10, with identical
code, it wins.

The reason: large `k` flattens the scoring curve, so rank 1 and rank 20 score almost
the same — which lets a weaker retriever's opinions drag documents down from the top
of a stronger retriever's list. Small `k` makes rank 1 count far more, protecting
precision at the top.

The weight sweep exposes the real trade-off too: weighting toward dense buys recall
(0.9517) at the cost of nDCG (0.7069). **Those two move in opposite directions — you
pick based on whether a reranker runs downstream**, because a reranker needs recall
and will resort the top anyway.

## Finding 2 — a reranker is only as good as its training distribution

`ms-marco-MiniLM-L-6-v2` is the default reranker in most RAG tutorials. Here it dropped
nDCG from 0.8101 to 0.7468.

It's trained on MS MARCO: short, conversational web-search queries against web
passages. SciFact is the opposite — the query is a formal scientific claim, the
documents are dense biomedical abstracts, and "relevant" means *this paper supports or
refutes the claim*, not *this paper is on the same topic*. The model has never seen
this distribution.

Swapping in `bge-reranker-base` (broader training data, 5× the parameters) recovered
most of the gap — 0.7468 → 0.7936 — which confirms the cause was the **model**, not the
pipeline. But it still lost to no reranking, at 9.7 seconds per query on CPU.

**Reranking is not a free upgrade.** On a corpus where first-stage retrieval is already
strong at the top and the notion of relevance is unusual, a general-purpose
cross-encoder is a downgrade you pay 300× latency for.

## Finding 3 — Recall@k is the ceiling on everything downstream

A reranker can only reorder what it is handed. Recall@50 here is 0.9443, so **94.4% is
the hard ceiling** on final accuracy no matter how good the reranker is.

This makes recall the number to optimise in stage one — stage-one nDCG matters much
less, since the reranker resorts the top anyway. It is also the answer to the most
common complaint in RAG work: *"my reranker isn't helping."* Usually it isn't a
reranker problem at all. The right documents never made it into the candidate set.

---

## Architecture

```
query
  ├─> BM25           (keyword — exact terms)     ──┐
  └─> dense vectors  (meaning — paraphrases)     ──┤
                                                   ├─> RRF fusion (k=10) ─> top 50
                                                   ┘                         │
                                             optional cross-encoder rerank <─┘
```

| file | what it does |
|---|---|
| `src/load.py` | BEIR corpus / queries / qrels loading |
| `src/sparse.py` | BM25 with stopword filtering |
| `src/dense.py` | bge-small embeddings, cosine search |
| `src/fusion.py` | weighted Reciprocal Rank Fusion |
| `src/rerank.py` | cross-encoder reranking |
| `src/metrics.py` | nDCG@k, Recall@k, MRR@k implemented from scratch |
| `run_ablation.py` | the full stage-by-stage experiment |
| `sweep_weights.py` | the RRF `k` and weight sweep |
| `compare_rerankers.py` | the reranker head-to-head |

Every module is commented to explain *why* it works that way, not just what it does.

## Running it

```bash
pip install -r requirements.txt
python download_data.py
python run_ablation.py --device cuda      # use cpu if you have no GPU
python sweep_weights.py
python compare_rerankers.py
```

Full ablation: ~2 minutes on an RTX 4060, ~26 minutes on CPU.

## Honest limitations

- **One dataset.** SciFact is small and unusual; the reranker finding may well reverse
  on a web-search-like corpus. The next experiment is the identical ablation on FiQA,
  to see whether MiniLM flips from −0.063 to a gain. That comparison turns *"reranking
  hurt"* into *"here is exactly when reranking helps."*
- **`k` was tuned on the test set.** In a real system that is leakage — tune on a dev split.
- **Metadata filtering** is implemented but not benchmarked (SciFact has no useful metadata).
- **No vector index.** Latency is single-query and unbatched, over brute-force dot
  products across 5,183 documents. At a million documents you need FAISS or HNSW and
  the latency picture changes completely.

## License

MIT
