# Documentation

A complete walkthrough of this project, written for someone learning to be an AI
engineer. It assumes you can read Python but **not** that you know anything about
retrieval, embeddings, or evaluation. Every term is defined where it first appears.

## Read in this order

| # | document | what you'll understand after it |
|---|---|---|
| 1 | [Why retrieval exists](01-why-retrieval-exists.md) | What RAG actually is, and why retrieval is the part that decides whether it works |
| 2 | [BM25: keyword search](02-bm25-keyword-search.md) | How keyword scoring works, with the formula worked by hand |
| 3 | [Embeddings: meaning search](03-embeddings-meaning-search.md) | What a vector actually is, cosine similarity, bi-encoders |
| 4 | [Fusion: combining retrievers](04-fusion-combining-retrievers.md) | Why you can't just add scores, and what RRF does instead |
| 5 | [Reranking: the accuracy pass](05-reranking-cross-encoders.md) | Cross-encoders vs bi-encoders, and why reranking is a bet not an upgrade |
| 6 | [Metrics: measuring retrieval](06-metrics-how-to-measure.md) | nDCG, Recall@k, MRR — each computed by hand on a tiny example |
| 7 | [Evaluation and leakage](07-evaluation-and-leakage.md) | Train/test splits, what leakage is, how it inflated this project by 2.2% |
| 8 | [Scaling: indexes and filters](08-scaling-indexes-and-filters.md) | ANN indexes, the recall/latency dial, and the metadata filtering trap |
| 9 | [Code walkthrough](09-code-walkthrough.md) | Every file and function in this repo, explained |
| 10 | [Running and extending](10-running-and-extending.md) | How to run it, add datasets, swap models, debug |
| 11 | [What this teaches you](11-what-this-teaches-you.md) | The transferable lessons, and the interview questions this prepares you for |

## The short version

If you read nothing else: this project builds a document search system four different
ways and measures each one against a human-labelled answer key. The measurement is
the project. Three of the four "standard" upgrades turned out to be neutral or
harmful on at least one dataset, and you would never know that without measuring.

That pattern — **assume nothing, measure everything, report what you find even when
it contradicts you** — is the actual skill. The retrieval knowledge is a bonus.
