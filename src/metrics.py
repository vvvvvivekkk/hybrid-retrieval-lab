"""The three numbers that tell you whether retrieval works.

nDCG@10  -- "normalised discounted cumulative gain at 10". The headline metric
            in retrieval papers, so it's the one to quote.
            * gain: each relevant doc in your top 10 earns points
            * discounted: points shrink the further down it sits (a correct
              answer at rank 1 is worth more than the same one at rank 9)
            * normalised: divided by the best score possible for that query, so
              0.0 = useless, 1.0 = perfect, and queries with different numbers
              of correct answers are comparable.

Recall@k -- of all correct documents, what fraction appeared anywhere in your
            top k? This is the metric that governs the reranker. A cross-encoder
            can only reorder what it's handed: if Recall@50 is 0.90, then 90% is
            the hard ceiling on your final accuracy no matter how good the
            reranker is. When people say "my reranker isn't helping", this is
            almost always why.

MRR@10   -- mean reciprocal rank: 1/(position of the first correct answer).
            Rank 1 -> 1.0, rank 4 -> 0.25. The metric to use when the user only
            ever looks at the top result.
"""
import math


def ndcg_at_k(ranked_ids, relevant: dict, k: int = 10) -> float:
    dcg = sum(relevant.get(d, 0) / math.log2(i + 1)
              for i, d in enumerate(ranked_ids[:k], start=1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, relevant: dict, k: int = 50) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranked_ids[:k]) & set(relevant))
    return hits / len(relevant)


def mrr_at_k(ranked_ids, relevant: dict, k: int = 10) -> float:
    for i, d in enumerate(ranked_ids[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def evaluate(run: dict, qrels: dict, ks=(10,), recall_ks=(50, 100)) -> dict:
    """run: query_id -> [doc_id, ...] ranked best-first."""
    out = {}
    for k in ks:
        out[f"nDCG@{k}"] = sum(ndcg_at_k(run[q], qrels[q], k) for q in run) / len(run)
        out[f"MRR@{k}"]  = sum(mrr_at_k(run[q],  qrels[q], k) for q in run) / len(run)
    for k in recall_ks:
        out[f"Recall@{k}"] = sum(recall_at_k(run[q], qrels[q], k) for q in run) / len(run)
    return out
