"""Limitation 3: metadata filtering was implemented but never benchmarked.

SciFact ships no real metadata, so inventing fake "departments" and claiming an
accuracy gain would be dishonest. What CAN be measured honestly is the mechanism --
post-filtering vs pre-filtering -- because the failure mode is structural and does
not depend on what the metadata means.

Setup: assign every document to a synthetic partition (deterministic hash), then
restrict search to a partition containing a given fraction of the corpus. The
partition labels are arbitrary by design; the point is what each filtering strategy
does as the filter narrows.

Expected, and the reason this matters in production: post-filtering returns fewer
and fewer results as the filter tightens, because the filter is applied AFTER the
top-k cut. Pre-filtering always returns a full k. Teams ship post-filtering because
it's one line of code, then get bug reports only from the users with narrow filters.
"""
import sys, json, hashlib, argparse, time; sys.path.insert(0, "src")
import numpy as np
from load import load_split
from dense import DenseRetriever
from metadata import apply_post_filter, build_mask, prefiltered_dense_search

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="scifact")
ap.add_argument("--device", default="cpu")
ap.add_argument("--top-k", type=int, default=20)
a = ap.parse_args()

corpus, queries, qrels = load_split(a.dataset, "test")
dense = DenseRetriever(corpus, device=a.device)

def partition(doc_id, frac):
    """Deterministic pseudo-random membership, stable across runs."""
    h = int(hashlib.md5(doc_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac

print(f"\ntop_k requested = {a.top_k}\n")
print(f"{'filter keeps':>14}{'post: avg results':>20}{'post: ms':>11}"
      f"{'pre: avg results':>19}{'pre: ms':>10}")
print("-" * 76)
rows = []
for frac in (1.0, 0.5, 0.2, 0.05, 0.01):
    allowed = {d for d in corpus if partition(d, frac)}
    mask = build_mask(dense.doc_ids, allowed)

    t = time.perf_counter(); post_n = []
    for q in queries.values():
        post_n.append(len(apply_post_filter(dense.search(q, a.top_k), allowed, a.top_k)))
    post_ms = (time.perf_counter() - t) / len(queries) * 1000

    t = time.perf_counter(); pre_n = []
    for q in queries.values():
        pre_n.append(len(prefiltered_dense_search(dense, q, mask, a.top_k)))
    pre_ms = (time.perf_counter() - t) / len(queries) * 1000

    r = {"filter_frac": frac, "allowed_docs": len(allowed),
         "post_avg_results": sum(post_n)/len(post_n), "post_ms": post_ms,
         "pre_avg_results": sum(pre_n)/len(pre_n), "pre_ms": pre_ms}
    rows.append(r)
    print(f"{f'{frac:.0%} ({len(allowed)})':>14}{r['post_avg_results']:>20.1f}"
          f"{post_ms:>11.1f}{r['pre_avg_results']:>19.1f}{pre_ms:>10.1f}")
print("-" * 76)
json.dump(rows, open(f"data/{a.dataset}/metadata_bench.json", "w"), indent=2)
