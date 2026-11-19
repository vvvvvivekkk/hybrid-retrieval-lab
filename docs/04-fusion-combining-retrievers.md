# 4. Fusion — combining two retrievers

**File in this repo:** `src/fusion.py`

## The problem nobody warns you about

You have BM25 and dense search. Both return ranked lists. Merge them. Easy?

```
BM25 results:              Dense results:
  doc_41   18.4              doc_41   0.81
  doc_7    12.1              doc_93   0.79
  doc_88    9.6              doc_7    0.77
```

**You cannot add these numbers.** BM25 scores are unbounded sums of IDF terms — they
can be 3 or 300 depending on query length and corpus. Cosine similarities are capped
in [−1, 1] and in practice cluster tightly between 0.6 and 0.9. Add them and BM25
drowns out dense entirely.

### Why normalising doesn't fix it

The obvious patch is min-max normalisation — rescale each list to [0, 1]:

```
normalised = (score − min) / (max − min)
```

This is fragile in a way that bites in production:

- **Per-query instability.** You normalise within each query's results. A query where
  the best BM25 score is 40 and a query where it's 4 both get rescaled to 1.0. You've
  erased the information that one query had a great match and the other had nothing.
- **Outlier sensitivity.** One document scoring 300 while everything else scores under
  20 squashes the entire rest of the list into near-zero.
- **It needs tuning you can't validate.** Now you need a weight α for
  `α × norm_bm25 + (1−α) × norm_dense`, tuned per corpus, and it drifts as your corpus
  changes.

## Reciprocal Rank Fusion

RRF sidesteps all of it by **throwing the scores away and using only positions**.

```
score(doc) = Σ        1 / (k + rank_in_that_list)
           lists
```

rank starts at 1. `k` is a constant, conventionally 60.

### Worked example (k = 60)

```
BM25:   1. doc_41    2. doc_7    3. doc_88
Dense:  1. doc_41    2. doc_93   3. doc_7

doc_41:  1/61 + 1/61  = 0.03279   ← top of both lists
doc_7:   1/62 + 1/63  = 0.03200   ← solid in both
doc_93:         1/62  = 0.01613   ← only one list
doc_88:  1/63         = 0.01587   ← only one list

Final: doc_41, doc_7, doc_93, doc_88
```

Documents both retrievers like rise. Documents only one likes still appear, but lower.
No scale problem, no normalisation, no per-query instability.

## What `k` actually does — and why the default is wrong

`k` is in the denominator, so it controls **how sharply early ranks are rewarded.**

| | rank 1 | rank 10 | rank 50 | rank 1 ÷ rank 50 |
|---|---|---|---|---|
| **k = 60** | 0.0164 | 0.0143 | 0.0091 | **1.8×** |
| **k = 10** | 0.0909 | 0.0500 | 0.0167 | **5.5×** |
| **k = 5** | 0.1667 | 0.0667 | 0.0182 | **9.2×** |

With **k=60**, rank 1 is worth only 1.8× rank 50. The curve is nearly flat, so a weak
retriever's mediocre opinions can drag down a strong retriever's top pick.

With **k=10**, rank 1 is worth 5.5× rank 50, so the top of each list dominates.

Measured on SciFact in this repo:

| rrf_k | nDCG@10 |
|---|---|
| 60 (the value everyone copies) | 0.7040 |
| 10 | **0.7188** |

**At k=60 the hybrid system scored *worse* than dense retrieval alone (0.7077).** The
"combine two retrievers" upgrade was actively harmful, purely because of a constant
copied from a 2009 paper that nobody questions.

This is the most useful habit in this whole project: **every default is a
hyperparameter until you've measured it.**

## Weights

`fusion.py` also supports weighting the lists:

```python
def rrf(result_lists, k=60, top_k=100, weights=None):
    fused[doc_id] += w / (k + rank)
```

Weights `[0.5, 1]` means "trust dense twice as much as BM25" — sensible here, since
dense scores higher alone on both datasets. Tuned honestly, `(0.5, 1)` won on **both**
SciFact and NFCorpus independently, which is mild evidence it generalises.

## The trade-off you must understand

Fusion moves two metrics in **opposite directions**:

| config | nDCG@10 | Recall@50 |
|---|---|---|
| dense alone | 0.7077 | 0.9250 |
| hybrid (k=10, 1:1) | **0.7188** | 0.9443 |
| hybrid (k=10, 0.5:1) | 0.7069 | **0.9517** |

**nDCG@10** = are the right docs at the very top?
**Recall@50** = are the right docs *anywhere* in the top 50?

Which you want depends on what consumes the output:

- **No reranker downstream?** Optimise nDCG. The top 5 go straight to the LLM.
- **Reranker downstream?** Optimise Recall@50. The reranker will resort the top
  anyway, and it can only work with what you hand it. See
  [doc 5](05-reranking-cross-encoders.md).

Most tutorials report one number and never mention the trade-off exists.

## Honest result

Tuned without leakage ([doc 7](07-evaluation-and-leakage.md)), hybrid's benefit
turned out to be **dataset-dependent**:

| dataset | dense alone | hybrid (honest) | verdict |
|---|---|---|---|
| SciFact | 0.7077 | 0.7069 | **ties** — wins only on recall (+2.7 pts) |
| NFCorpus | 0.3408 | **0.3562** | **wins** (+4.5%) |

"Hybrid search is better" is not a law. It's a hypothesis you test per corpus.

→ Next: [Reranking: the accuracy pass](05-reranking-cross-encoders.md)
