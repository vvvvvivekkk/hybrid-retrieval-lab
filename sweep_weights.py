"""Why did hybrid lose to dense? Test the obvious suspect: equal weighting.

RRF with weights [1,1] treats both retrievers as equally trustworthy. On this
dataset they are NOT -- dense scores 0.708 and BM25 scores 0.665. When you fuse
a strong list with a weaker one at equal weight, the weak list drags documents
down from the top of the strong list. Recall improves (two nets catch more fish)
but precision at rank 1-10 suffers.

So: sweep the weight and watch nDCG and Recall move in opposite directions.
"""
import sys, json; sys.path.insert(0, "src")
from load import load_all
from fusion import rrf
from metrics import evaluate

corpus, queries, qrels = load_all()
cache = json.load(open("data/runs_cache.json"))

print(f"{'w_sparse':>9}{'w_dense':>9}{'rrf_k':>7}{'nDCG@10':>10}{'Recall@50':>11}{'Recall@100':>12}")
print("-" * 58)
rows = []
for k in (10, 60):
    for ws, wd in [(1,0),(0,1),(1,1),(0.5,1),(0.3,1),(0.2,1)]:
        run = {}
        for qid in queries:
            c = cache[qid]
            lists, wts = [], []
            if ws: lists.append(c["sparse"]); wts.append(ws)
            if wd: lists.append(c["dense"]);  wts.append(wd)
            run[qid] = [d for d, _ in rrf(lists, k=k, top_k=100, weights=wts)]
        m = evaluate(run, qrels, ks=(10,), recall_ks=(50, 100))
        rows.append({"rrf_k": k, "w_sparse": ws, "w_dense": wd, **m})
        print(f"{ws:>9}{wd:>9}{k:>7}{m['nDCG@10']:>10.4f}{m['Recall@50']:>11.4f}{m['Recall@100']:>12.4f}")
json.dump(rows, open("sweep_results.json", "w"), indent=2)
