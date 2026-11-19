"""Why did the reranker HURT?

Hypothesis: domain mismatch. ms-marco-MiniLM-L-6-v2 is trained on MS MARCO --
short everyday web-search queries ("how tall is the eiffel tower") against web
passages. SciFact is the opposite: the "query" is a formal scientific claim and
the documents are dense biomedical abstracts. A cross-encoder is only as good as
the query/document distribution it was trained on, and this one has never seen
anything like this.

If that's right, a reranker trained on broader/scientific data should beat it on
the SAME candidates. If it's wrong, both rerankers fail and the real problem is
my pipeline.
"""
import sys, json, time; sys.path.insert(0, "src")
from load import load_all
from fusion import rrf
from rerank import Reranker
from metrics import evaluate

N_Q, N_CAND = 80, 25
corpus, queries, qrels = load_all()
cache = json.load(open("data/runs_cache.json"))
qids = list(queries)[:N_Q]

fused = {q: rrf([cache[q]["sparse"], cache[q]["dense"]], k=10, top_k=100) for q in qids}
sub_qrels = {q: qrels[q] for q in qids}
base = {q: [d for d, _ in fused[q]] for q in qids}
print(f"subset: {N_Q} queries, reranking top {N_CAND}")
m = evaluate(base, sub_qrels, ks=(10,), recall_ks=(25,))
print(f"{'no rerank (hybrid k=10)':<34}{m['nDCG@10']:>9.4f}{m['MRR@10']:>9.4f}      --", flush=True)

for name in ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"]:
    rr = Reranker(name, device="cpu", batch_size=32)
    t0 = time.perf_counter(); run = {}
    for q in qids:
        head = [d for d, _ in rr.rerank(queries[q], fused[q][:N_CAND], corpus, top_k=N_CAND)]
        run[q] = head + [d for d, _ in fused[q] if d not in set(head)]
    ms = (time.perf_counter() - t0) / len(qids) * 1000
    m = evaluate(run, sub_qrels, ks=(10,), recall_ks=(25,))
    print(f"{name.split('/')[-1]:<34}{m['nDCG@10']:>9.4f}{m['MRR@10']:>9.4f}{ms:>9.0f}ms", flush=True)
