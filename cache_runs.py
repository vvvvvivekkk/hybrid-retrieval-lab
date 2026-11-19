"""Cache the expensive parts so experiments afterwards are instant."""
import sys, json, time; sys.path.insert(0, "src")
import numpy as np
from load import load_all
from sparse import SparseRetriever
from dense import DenseRetriever

corpus, queries, qrels = load_all()
sparse = SparseRetriever(corpus)
t = time.perf_counter(); dense = DenseRetriever(corpus, device="cpu")
print(f"dense idx {time.perf_counter()-t:.0f}s", flush=True)

cache = {}
for i, (qid, qt) in enumerate(queries.items(), 1):
    cache[qid] = {"sparse": sparse.search(qt, 100), "dense": dense.search(qt, 100)}
    if i % 100 == 0: print(f"{i}/{len(queries)}", flush=True)
json.dump(cache, open("data/runs_cache.json", "w"))
print("cached ->", len(cache))
