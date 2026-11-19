"""Diagnostic sweep: how do rrf_k and the fusion weights trade off?

This is an EXPLORATION tool, not an evaluation tool. It shows the shape of the
trade-off across a split. To choose a config honestly, use tune.py, which fits on
train/validation and reports on test.
"""
import sys, json, argparse; sys.path.insert(0, "src")
from load import load_qrels
from fusion import rrf
from metrics import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="scifact")
ap.add_argument("--split", default="test")
a = ap.parse_args()

blob = json.load(open(f"data/{a.dataset}/runs_cache.json"))
cache, qids = blob["runs"], blob["splits"][a.split]
qrels = load_qrels(a.dataset, a.split)

print(f"{a.dataset} / {a.split} ({len(qids)} queries)\n")
print(f"{'w_sparse':>9}{'w_dense':>9}{'rrf_k':>7}{'nDCG@10':>10}{'Recall@50':>11}{'Recall@100':>12}")
print("-" * 58)
rows = []
for k in (5, 10, 20, 60):
    for ws, wd in [(1, 0), (0, 1), (1, 1), (0.5, 1), (0.3, 1)]:
        run = {}
        for qid in qids:
            c = cache[qid]
            lists = [l for l, w in ((c["sparse"], ws), (c["dense"], wd)) if w]
            wts = [w for w in (ws, wd) if w]
            run[qid] = [d for d, _ in rrf(lists, k=k, top_k=100, weights=wts)]
        m = evaluate(run, {q: qrels[q] for q in run}, ks=(10,), recall_ks=(50, 100))
        rows.append({"rrf_k": k, "w_sparse": ws, "w_dense": wd, **m})
        print(f"{ws:>9}{wd:>9}{k:>7}{m['nDCG@10']:>10.4f}{m['Recall@50']:>11.4f}{m['Recall@100']:>12.4f}")
json.dump(rows, open(f"data/{a.dataset}/sweep_results.json", "w"), indent=2)
