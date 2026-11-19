"""The experiment: add one component at a time, measure what each one bought you.

This file is the actual portfolio deliverable. Anyone can wire up a retriever;
the thing that reads as engineering is the ablation table at the end -- proof
that you measured each piece instead of assuming it helped.
"""
import sys, time, json, argparse
sys.path.insert(0, "src")

from load import load_all
from sparse import SparseRetriever
from dense import DenseRetriever
from fusion import rrf
from rerank import Reranker
from metrics import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--device", default=None, help="cuda for your 4060, cpu otherwise")
ap.add_argument("--candidates", type=int, default=50, help="how many docs reach the reranker")
ap.add_argument("--rrf-k", type=int, default=60)
ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
ap.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
args = ap.parse_args()

corpus, queries, qrels = load_all()
print(f"{len(corpus)} docs | {len(queries)} queries\n")

t0 = time.perf_counter(); sparse = SparseRetriever(corpus)
print(f"BM25 index built in {time.perf_counter()-t0:.1f}s")
t0 = time.perf_counter(); dense = DenseRetriever(corpus, args.embed_model, device=args.device)
print(f"Dense index built in {time.perf_counter()-t0:.1f}s")
reranker = Reranker(args.rerank_model, device=args.device)

runs = {"bm25": {}, "dense": {}, "hybrid": {}, "hybrid+rerank": {}}
lat  = {k: [] for k in runs}

for n, (qid, qtext) in enumerate(queries.items(), 1):
    t = time.perf_counter(); s_res = sparse.search(qtext, 100); t_s = time.perf_counter() - t
    t = time.perf_counter(); d_res = dense.search(qtext, 100);  t_d = time.perf_counter() - t
    t = time.perf_counter(); fused = rrf([s_res, d_res], k=args.rrf_k, top_k=100); t_f = time.perf_counter() - t
    t = time.perf_counter()
    reranked = reranker.rerank(qtext, fused[:args.candidates], corpus, top_k=args.candidates)
    t_r = time.perf_counter() - t

    runs["bm25"][qid]          = [d for d, _ in s_res]
    runs["dense"][qid]         = [d for d, _ in d_res]
    runs["hybrid"][qid]        = [d for d, _ in fused]
    # The reranker only reorders the top `candidates`; everything below keeps its
    # fusion order. Without this tail the Recall@100 column would be a lie.
    head = [d for d, _ in reranked]
    runs["hybrid+rerank"][qid] = head + [d for d, _ in fused if d not in set(head)]

    lat["bm25"].append(t_s * 1000)
    lat["dense"].append(t_d * 1000)
    lat["hybrid"].append((t_s + t_d + t_f) * 1000)
    lat["hybrid+rerank"].append((t_s + t_d + t_f + t_r) * 1000)
    if n % 50 == 0:
        print(f"  {n}/{len(queries)} queries")

print("\n" + "=" * 78)
print(f"{'stage':<16}{'nDCG@10':>10}{'MRR@10':>10}{'Recall@50':>12}{'Recall@100':>12}{'p50 ms':>10}{'p95 ms':>9}")
print("-" * 78)
results = {}
for name in runs:
    m = evaluate(runs[name], qrels, ks=(10,), recall_ks=(50, 100))
    L = sorted(lat[name])
    p50, p95 = L[len(L) // 2], L[int(len(L) * 0.95)]
    results[name] = {**m, "p50_ms": p50, "p95_ms": p95}
    print(f"{name:<16}{m['nDCG@10']:>10.4f}{m['MRR@10']:>10.4f}"
          f"{m['Recall@50']:>12.4f}{m['Recall@100']:>12.4f}{p50:>10.1f}{p95:>9.1f}")
print("=" * 78)

json.dump(results, open("results.json", "w"), indent=2)
print("\nsaved -> results.json")
