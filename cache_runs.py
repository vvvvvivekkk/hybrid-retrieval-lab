"""Index once, cache the per-query result lists, so fusion/tuning experiments are instant.

The embedding pass is by far the most expensive step (minutes on CPU). Everything
downstream -- RRF weights, the k constant, reranker choice -- only needs the ranked
lists, not the model. So we pay once and cache.
"""
import sys, json, time, argparse; sys.path.insert(0, "src")
from load import load_corpus, load_queries, load_qrels, available_splits
from sparse import SparseRetriever
from dense import DenseRetriever

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="scifact")
ap.add_argument("--device", default="cpu")
ap.add_argument("--max-train", type=int, default=250, help="cap train queries (tuning needs far fewer than eval)")
a = ap.parse_args()

corpus = load_corpus(a.dataset)
all_q = load_queries(a.dataset)
splits = [s for s in available_splits(a.dataset) if s in ("train", "validation", "test")]
print(f"{a.dataset}: {len(corpus)} docs | splits {splits}", flush=True)

wanted = {}
for s in splits:
    qr = load_qrels(a.dataset, s)
    qids = [q for q in all_q if q in qr]
    if s != "test":
        qids = qids[:a.max_train]
    wanted[s] = qids
    print(f"  {s}: {len(qids)} queries", flush=True)

sparse = SparseRetriever(corpus)
t = time.perf_counter(); dense = DenseRetriever(corpus, device=a.device)
print(f"dense index: {time.perf_counter()-t:.0f}s", flush=True)

cache = {}
todo = sorted({q for v in wanted.values() for q in v})
for i, qid in enumerate(todo, 1):
    qt = all_q[qid]
    cache[qid] = {"sparse": sparse.search(qt, 100), "dense": dense.search(qt, 100)}
    if i % 100 == 0: print(f"  {i}/{len(todo)}", flush=True)

json.dump({"splits": wanted, "runs": cache},
          open(f"data/{a.dataset}/runs_cache.json", "w"))
print(f"cached {len(cache)} queries -> data/{a.dataset}/runs_cache.json")
