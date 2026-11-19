"""Approximate Nearest Neighbour search -- making dense retrieval scale.

The brute-force dense retriever compares the query vector against EVERY document
vector. That's exact, and it's fine at 5,000 documents. At 5 million it is not:
cost grows linearly, so you'd be doing 5 million dot products per query.

An ANN index trades a little accuracy for a lot of speed. The one used here is
HNSW -- "Hierarchical Navigable Small World". The idea: build a graph where each
document vector is a node linked to its nearest neighbours, with a few sparse
"express lane" layers on top. To search, you drop in at the top layer, greedily
walk toward the query, and descend. You touch a few hundred nodes instead of
millions.

The cost is that it's *approximate* -- it can miss a true nearest neighbour. So
the honest way to report an ANN index is not "it's faster", but this pair:

    recall vs brute force   (what fraction of the true top-k did it find?)
    latency                 (what did that buy?)

`efSearch` is the dial between them: higher = explores more of the graph = more
accurate and slower.

Measured in this repo (5,183 docs, 384 dims, top-100):

    flat (exact)      recall 1.000    0.339 ms
    hnsw ef=16        recall 0.691    0.082 ms     4.6x faster, useless recall
    hnsw ef=128       recall 0.990    0.183 ms     1.9x faster, usable
    hnsw ef=256       recall 0.999    0.304 ms     1.1x faster, barely worth it

So the speedup is real but you buy it with recall, and at this corpus size both
options are a fraction of a millisecond -- irrelevant next to the 21 ms the query
embedding costs. ANN earns its place at millions of documents, not thousands.
Shipping it here would be cargo-culting, which is the point of measuring first.
"""
import numpy as np
import faiss


class FlatIndex:
    """Exact brute-force search. The ground truth everything else is measured against."""
    name = "flat (exact)"

    def __init__(self, emb: np.ndarray):
        self.index = faiss.IndexFlatIP(emb.shape[1])   # IP = inner product
        self.index.add(emb)                            # vectors are L2-normalised,
                                                       # so inner product == cosine
    def search(self, q: np.ndarray, k: int):
        s, i = self.index.search(q.reshape(1, -1), k)
        return i[0], s[0]


class HNSWIndex:
    """Approximate graph-based search."""

    def __init__(self, emb: np.ndarray, M: int = 32, ef_construction: int = 200,
                 ef_search: int = 64):
        # M = how many neighbour links each node keeps. Higher = better recall,
        # more memory, slower build.
        self.index = faiss.IndexHNSWFlat(emb.shape[1], M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self.index.add(emb)
        self.ef_search = ef_search
        self.name = f"hnsw (M={M}, ef={ef_search})"

    def set_ef(self, ef: int):
        self.index.hnsw.efSearch = ef
        self.ef_search = ef
        self.name = f"hnsw (M={self.index.hnsw.nb_neighbors(0) if False else 32}, ef={ef})"

    def search(self, q: np.ndarray, k: int):
        s, i = self.index.search(q.reshape(1, -1), k)
        return i[0], s[0]


def recall_vs_exact(approx_ids, exact_ids, k: int) -> float:
    """What fraction of the exact top-k did the approximate index actually return?"""
    return len(set(approx_ids[:k]) & set(exact_ids[:k])) / k
