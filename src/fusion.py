"""Reciprocal Rank Fusion -- merging two ranked lists into one.

The problem: BM25 returns scores like 18.4 and the dense retriever returns
scores like 0.81. They're on completely different scales, so you cannot add
them. Normalising them (min-max, z-score) is fragile -- one outlier query
wrecks the scaling.

RRF sidesteps the whole issue by throwing the scores away and using only the
*positions*:

    score(doc) = sum over each list of  1 / (k + rank_in_that_list)

with rank starting at 1 and k = 60 by convention. A document at rank 1 in both
lists scores 2/61. A document at rank 1 in one list and absent from the other
scores 1/61 -- still strong. Documents both lists rank highly float to the top.

k controls how sharply early ranks are favoured. Small k = top positions
dominate. Large k = flatter, deeper agreement matters more. 60 is the value from
the original 2009 paper and is a genuinely good default.
"""
from collections import defaultdict


def rrf(result_lists, k: int = 60, top_k: int = 100, weights=None):
    """result_lists: list of [(doc_id, score), ...], each already sorted best-first."""
    weights = weights or [1.0] * len(result_lists)
    fused = defaultdict(float)
    for results, w in zip(result_lists, weights):
        for rank, (doc_id, _score) in enumerate(results, start=1):
            fused[doc_id] += w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:top_k]
