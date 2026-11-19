"""Metadata filtering -- and the trap inside it.

Real corpora carry structured fields: date, department, language, access level.
Users want "search only 2024 docs" or "only documents this user may see". There
are two ways to do it, and the difference is a classic production bug.

  POST-filter: retrieve top-k as normal, then drop whatever fails the filter.
      Simple, works with any index. But if only 5% of documents match the filter,
      your top-50 might contain 2 matching documents -- or zero. You asked for 50
      results and got 2. Recall collapses, silently, and only for users with
      narrow filters.

  PRE-filter: restrict the candidate set BEFORE scoring, so you always return a
      full k of matching documents.
      Correct recall. Costs more: with brute force it's a cheap mask, but with an
      ANN graph index it is genuinely hard (the graph's edges may all lead out of
      your allowed set), which is why vector databases advertise "filtered search"
      as a feature rather than a given.

This module implements both so the difference can be measured, not argued about.
"""
import numpy as np


def apply_post_filter(results, allowed: set, top_k: int):
    """Filter after retrieval. Returns fewer than top_k when the filter is narrow."""
    return [(d, s) for d, s in results if d in allowed][:top_k]


def build_mask(doc_ids, allowed: set) -> np.ndarray:
    """Boolean mask over the document list, for pre-filtered scoring."""
    return np.fromiter((d in allowed for d in doc_ids), dtype=bool, count=len(doc_ids))


def prefiltered_dense_search(dense, query: str, mask: np.ndarray, top_k: int):
    """Score ONLY the documents allowed by the mask, so top_k is always full."""
    from dense import QUERY_PREFIX
    q = dense.model.encode([QUERY_PREFIX + query], convert_to_numpy=True,
                           normalize_embeddings=True).astype(np.float32)[0]
    idx_allowed = np.flatnonzero(mask)
    if idx_allowed.size == 0:
        return []
    scores = dense.emb[idx_allowed] @ q
    k = min(top_k, idx_allowed.size)
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[(-scores[top]).argsort()]
    return [(dense.doc_ids[idx_allowed[i]], float(scores[i])) for i in top]
