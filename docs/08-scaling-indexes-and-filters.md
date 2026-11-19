# 8. Scaling — indexes and filters

**Files in this repo:** `src/ann.py`, `src/metadata.py`

Two things that are invisible at 5,000 documents and decide whether your system works
at 5 million.

---

## Part 1: Approximate Nearest Neighbour indexes

### The problem

`dense.py` does a **brute-force** search: compare the query vector against every
document vector.

```python
scores = self.emb @ q     # (5183, 384) @ (384,) → 5183 scores
```

Exact, simple, and **linear in corpus size**. 5,183 documents takes 0.34 ms.
5 million would take roughly 330 ms — per query, before you've done anything else.

### HNSW

**Hierarchical Navigable Small World.** The idea:

1. Build a graph where each document vector is a node connected to its nearest
   neighbours.
2. Add sparse upper layers — "express lanes" with long-range links.
3. To search: enter at the top layer, greedily walk toward the query, descend a layer,
   repeat.

You touch a few hundred nodes instead of millions. The cost is that it's
**approximate** — the greedy walk can get stuck in a local neighbourhood and miss a
true nearest neighbour.

### Measuring it honestly

The wrong way to report an ANN index is "it's 4× faster." The honest way is the
**pair** — because speed without recall is meaningless:

| index | recall@100 vs exact | ms/query | build |
|---|---|---|---|
| flat (exact) | 1.000 | 0.339 | 0 s |
| hnsw ef=16 | 0.691 | 0.082 | 1.7 s |
| hnsw ef=32 | 0.851 | 0.074 | 1.8 s |
| hnsw ef=64 | 0.951 | 0.176 | 2.0 s |
| hnsw ef=128 | 0.990 | 0.183 | 2.0 s |
| hnsw ef=256 | 0.999 | 0.304 | 1.7 s |

**Recall vs exact** = what fraction of the true top-100 did the approximate index find?

`efSearch` is the dial: how many candidates the graph walk keeps alive. Higher = more
of the graph explored = more accurate, slower.

### What this actually says

The headline "4.6× faster" is at **ef=16, recall 0.691** — you lost a third of your
correct documents. Useless.

At usable recall (ef=128, 0.990) it's **1.9×**. And both numbers are a fraction of a
millisecond — irrelevant next to the **21 ms** the query embedding already costs. The
bottleneck isn't the search.

**Conclusion: do not ship an ANN index at 5,000 documents.** It adds a dependency,
build time, memory, and a recall penalty, to optimise something that isn't your
bottleneck. It earns its place at millions of documents, where brute force genuinely
breaks.

Knowing *when not* to use the sophisticated tool is a senior-engineer signal. Adding
FAISS because a blog post said to is the opposite.

### Other index types you'll meet

- **IVF** (inverted file) — cluster vectors, search only the nearest clusters. Lower
  memory than HNSW, needs a training step.
- **PQ** (product quantization) — compress vectors to a few bytes each. Big memory
  savings, further recall loss. Combined as **IVF-PQ** for billion-scale.
- **Flat** — exact brute force. Correct default until proven otherwise.

---

## Part 2: Metadata filtering, and the trap

Real corpora have structured fields: date, department, language, access level. Users
want "only 2024 documents" or — critically — "only documents this user may see."

There are two ways to do it. **One of them is broken, and it's the one everybody
ships**, because it's one line of code.

### Post-filtering (the trap)

```python
results = retriever.search(query, top_k=20)
results = [r for r in results if r.id in allowed]
```

Retrieve the top 20, then drop what fails the filter. Works with any index. Obvious.

The failure: the filter is applied **after** the top-k cut. If only 5% of your corpus
matches, most of your 20 results get deleted and nothing replaces them.

### Pre-filtering (the fix)

Restrict the candidate set **before** scoring, so the top-k is always full:

```python
idx_allowed = np.flatnonzero(mask)
scores = dense.emb[idx_allowed] @ q     # score only permitted documents
```

### Measured

Requesting **top-20** on SciFact, with synthetic partitions:

| filter keeps | post-filter returns | pre-filter returns |
|---|---|---|
| 100% | 20.0 | 20.0 |
| 50% | 9.9 | 20.0 |
| 20% | 3.9 | 20.0 |
| **5%** | **0.9** | 20.0 |
| **1%** | **0.2** | 20.0 |

At a 5% filter, post-filtering returns **0.9 results when you asked for 20.** At 1%,
most queries return **nothing at all.**

### Why this survives testing

This is the part worth internalising. The bug is **invisible to the people who test
the system.**

Developers and QA test with broad filters or none — 20 results come back, everything
looks fine. Only users with narrow filters hit it: the user in a small department, the
user querying a rare date range, the user whose permissions grant access to 2% of
documents. They get an empty page, conclude "search is broken for me," and often don't
report it.

If your filter is a **permission check**, the users most affected are the ones with
the least access — which is usually the most sensitive deployment you have.

### Why anyone post-filters anyway

Pre-filtering is genuinely harder with an ANN index. The HNSW graph's edges were built
over the *whole* corpus; if you restrict to 1% of nodes, the walk's neighbours mostly
lead outside the allowed set and the search degenerates. This is why vector databases
advertise "filtered vector search" as a feature rather than a given — Qdrant, Weaviate
and pgvector each solve it differently, and the differences matter.

With brute force it's trivial, as above: a boolean mask. One more reason not to reach
for an ANN index before you need it.

→ Next: [Code walkthrough](09-code-walkthrough.md)
