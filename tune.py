"""Fixing the leakage: tune on TRAIN, report on TEST.

The original version of this repo swept the RRF constant `k` across the test set and
reported the best value it found. That is test-set leakage. The number it produced
(nDCG@10 = 0.7188) describes how well the config fitted that particular answer key --
not how the system performs on queries it has never seen.

The honest protocol, and the one used here:

  1. Grid-search every config on the TRAIN split.
  2. Pick the single best config by train score. Freeze it.
  3. Run that one frozen config on TEST. Report that number and only that number.

This script also measures the leakage itself, by printing the best-on-test score
alongside. The gap between them is how much the original number was inflated --
which is worth knowing, because that same gap shows up in any project that tunes
on its evaluation set.
"""
import sys, json, argparse, itertools; sys.path.insert(0, "src")
from load import load_corpus, load_queries, load_qrels
from fusion import rrf
from metrics import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="scifact")
ap.add_argument("--tune-split", default=None, help="defaults to validation if present else train")
a = ap.parse_args()

blob = json.load(open(f"data/{a.dataset}/runs_cache.json"))
cache, splits = blob["runs"], blob["splits"]
tune_split = a.tune_split or ("validation" if "validation" in splits else "train")
print(f"dataset={a.dataset}  tune on '{tune_split}' ({len(splits[tune_split])} q)  "
      f"report on 'test' ({len(splits['test'])} q)\n")

qrels = {s: load_qrels(a.dataset, s) for s in splits}

GRID = list(itertools.product([5, 10, 20, 30, 60, 100],          # rrf_k
                              [(1,1), (0.5,1), (0.3,1), (1,0.5)]))  # weights

def score(split, rrf_k, ws):
    run = {}
    for qid in splits[split]:
        c = cache[qid]
        run[qid] = [d for d, _ in rrf([c["sparse"], c["dense"]], k=rrf_k,
                                      top_k=100, weights=list(ws))]
    return evaluate(run, {q: qrels[split][q] for q in run}, ks=(10,), recall_ks=(50,))

# ---- step 1+2: search the grid on the tuning split, freeze the winner ----
tuned = [(score(tune_split, k, w)["nDCG@10"], k, w) for k, w in GRID]
tuned.sort(reverse=True)
best_train, bk, bw = tuned[0]
print(f"best on {tune_split}:  rrf_k={bk}  weights={bw}  -> nDCG@10 {best_train:.4f}")

# ---- step 3: that frozen config, on test ----
honest = score("test", bk, bw)

# ---- and, for comparison only, the leaky number ----
leaky = max((score("test", k, w)["nDCG@10"], k, w) for k, w in GRID)

# baselines
base_dense  = score("test", 60, (0, 1))
base_sparse = score("test", 60, (1, 0))

print(f"\n{'config':<44}{'nDCG@10':>10}{'Recall@50':>12}")
print("-" * 66)
print(f"{'BM25 only':<44}{base_sparse['nDCG@10']:>10.4f}{base_sparse['Recall@50']:>12.4f}")
print(f"{'dense only':<44}{base_dense['nDCG@10']:>10.4f}{base_dense['Recall@50']:>12.4f}")
print(f"{f'hybrid, tuned on {tune_split} (k={bk}, w={bw})':<44}"
      f"{honest['nDCG@10']:>10.4f}{honest['Recall@50']:>12.4f}   <- HONEST")
print(f"{f'hybrid, best-on-test (k={leaky[1]}, w={leaky[2]})':<44}{leaky[0]:>10.4f}"
      f"{'':>12}   <- leaked")
print("-" * 66)
print(f"leakage inflation: {leaky[0]-honest['nDCG@10']:+.4f} nDCG@10 "
      f"({(leaky[0]/honest['nDCG@10']-1)*100:+.2f}%)")

json.dump({"dataset": a.dataset, "tune_split": tune_split,
           "chosen": {"rrf_k": bk, "weights": list(bw)},
           "test_honest": honest, "test_leaked_best": leaky[0],
           "baseline_dense": base_dense, "baseline_sparse": base_sparse},
          open(f"data/{a.dataset}/tuned.json", "w"), indent=2)
