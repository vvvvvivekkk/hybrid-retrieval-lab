"""Limitation 4: no vector index. Is brute force actually a problem here?

Measures the honest pair for an approximate index -- recall against exact search,
and the latency it bought -- instead of assuming ANN is a free win.
"""
import sys, json, time, argparse; sys.path.insert(0, "src")
import numpy as np
sys.path.insert(0, "src")
from load import load_split
from dense import DenseRetriever, QUERY_PREFIX
from ann import FlatIndex, HNSWIndex, recall_vs_exact

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="scifact")
ap.add_argument("--device", default="cpu")
ap.add_argument("--k", type=int, default=100)
a = ap.parse_args()

corpus, queries, _ = load_split(a.dataset, "test")
dense = DenseRetriever(corpus, device=a.device)
emb = dense.emb
qvecs = dense.model.encode([QUERY_PREFIX + t for t in queries.values()],
                           convert_to_numpy=True, normalize_embeddings=True,
                           batch_size=64).astype(np.float32)
print(f"\n{len(corpus)} docs x {emb.shape[1]} dims | {len(qvecs)} queries | top-{a.k}\n")

def bench(idx, qv, k):
    outs, t0 = [], time.perf_counter()
    for q in qv:
        ids, _ = idx.search(q, k)
        outs.append(ids)
    return outs, (time.perf_counter() - t0) / len(qv) * 1000

flat = FlatIndex(emb)
exact, flat_ms = bench(flat, qvecs, a.k)

rows = [{"index": "flat (exact)", "recall": 1.0, "ms": flat_ms, "build_s": 0}]
print(f"{'index':<26}{'recall@100 vs exact':>22}{'ms/query':>11}{'build s':>10}")
print("-" * 69)
print(f"{'flat (exact)':<26}{1.0:>22.4f}{flat_ms:>11.3f}{0:>10.1f}")

for ef in (16, 32, 64, 128, 256):
    t = time.perf_counter(); h = HNSWIndex(emb, M=32, ef_search=ef); build = time.perf_counter() - t
    approx, ms = bench(h, qvecs, a.k)
    rec = sum(recall_vs_exact(ap_, ex, a.k) for ap_, ex in zip(approx, exact)) / len(exact)
    rows.append({"index": f"hnsw ef={ef}", "recall": rec, "ms": ms, "build_s": build})
    print(f"{f'hnsw (M=32, ef={ef})':<26}{rec:>22.4f}{ms:>11.3f}{build:>10.1f}")
print("-" * 69)
fastest = min(r["ms"] for r in rows if r["index"] != "flat (exact)")
print(f"\nbrute force: {flat_ms:.3f} ms/query | best HNSW: {fastest:.3f} ms/query "
      f"-> speedup {flat_ms/fastest:.2f}x")
json.dump(rows, open(f"data/{a.dataset}/ann_bench.json", "w"), indent=2)
