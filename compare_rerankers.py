"""Does reranking help? It depends entirely on the corpus.

ms-marco-MiniLM-L-6-v2 is the default reranker in most RAG tutorials. It is trained
on MS MARCO: short, conversational web-search queries against web passages.

  SciFact  queries are formal scientific claims; "relevant" means a paper SUPPORTS
           or REFUTES the claim. Nothing like MS MARCO.
  NFCorpus queries are real search queries typed by real people on a nutrition site.
           Much closer to MS MARCO.

If the training-distribution theory is right, the same reranker should hurt on the
first and help on the second. Run this on both and find out.
"""
import sys, json, time, argparse; sys.path.insert(0, "src")
from load import load_corpus, load_queries, load_qrels
from fusion import rrf
from rerank import Reranker
from metrics import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="scifact")
ap.add_argument("--device", default="cpu")
ap.add_argument("--queries", type=int, default=80)
ap.add_argument("--candidates", type=int, default=25)
ap.add_argument("--models", nargs="+",
                default=["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"])
a = ap.parse_args()

corpus = load_corpus(a.dataset)
blob = json.load(open(f"data/{a.dataset}/runs_cache.json"))
cache = blob["runs"]
qrels = load_qrels(a.dataset, "test")
all_q = load_queries(a.dataset)
qids = [q for q in blob["splits"]["test"]][:a.queries]

# the tuned fusion config, from tune.py -- not a number picked on the test set
tuned = json.load(open(f"data/{a.dataset}/tuned.json"))["chosen"]
fused = {q: rrf([cache[q]["sparse"], cache[q]["dense"]],
                k=tuned["rrf_k"], top_k=100, weights=tuned["weights"]) for q in qids}
sub = {q: qrels[q] for q in qids}

print(f"\n{a.dataset}: {len(qids)} queries, reranking top {a.candidates} "
      f"(fusion k={tuned['rrf_k']}, w={tuned['weights']})\n")
print(f"{'model':<36}{'nDCG@10':>9}{'MRR@10':>9}{'delta':>9}{'ms/q':>9}")
print("-" * 72)

base = evaluate({q: [d for d, _ in fused[q]] for q in qids}, sub, ks=(10,), recall_ks=(25,))
b = base["nDCG@10"]
print(f"{'no rerank (baseline)':<36}{b:>9.4f}{base['MRR@10']:>9.4f}{'--':>9}{'--':>9}")

rows = [{"model": "none", "ndcg": b, "mrr": base["MRR@10"], "ms": 0}]
for name in a.models:
    rr = Reranker(name, device=a.device, batch_size=32)
    t0 = time.perf_counter(); run = {}
    for q in qids:
        head = [d for d, _ in rr.rerank(all_q[q], fused[q][:a.candidates], corpus,
                                        top_k=a.candidates)]
        run[q] = head + [d for d, _ in fused[q] if d not in set(head)]
    ms = (time.perf_counter() - t0) / len(qids) * 1000
    m = evaluate(run, sub, ks=(10,), recall_ks=(25,))
    rows.append({"model": name, "ndcg": m["nDCG@10"], "mrr": m["MRR@10"], "ms": ms})
    print(f"{name.split('/')[-1]:<36}{m['nDCG@10']:>9.4f}{m['MRR@10']:>9.4f}"
          f"{m['nDCG@10']-b:>+9.4f}{ms:>9.0f}", flush=True)
print("-" * 72)
json.dump(rows, open(f"data/{a.dataset}/rerank_bench.json", "w"), indent=2)
