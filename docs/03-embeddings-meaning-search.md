# 3. Embeddings — meaning search

**File in this repo:** `src/dense.py`

## What a vector actually is

An **embedding** is a list of numbers representing a piece of text. In this project,
384 of them:

```
"milk causes acne"  →  [0.021, -0.145, 0.882, ..., 0.043]   (384 numbers)
```

Those numbers aren't arbitrary. The model producing them was trained so that **texts
with similar meaning produce similar number lists**. "milk causes acne" and "dairy
leads to breakouts" land close together despite sharing no words.

Think of each number as a coordinate. Two numbers = a point on a page. Three = a point
in a room. 384 = a point in a space you can't picture but the maths handles fine.
Similar meanings cluster; unrelated meanings sit far apart.

## Measuring "close"

The standard measure is **cosine similarity** — the angle between two vectors,
ignoring their length.

```
cos(A, B) = (A · B) / (|A| × |B|)
```

Range −1 to 1: **1** = same direction (same meaning), **0** = unrelated, **−1** =
opposite.

Angle rather than distance, because length usually encodes something you don't care
about (roughly, text length and word count). Two documents about the same topic should
match whether one is a sentence or a page.

**The trick this repo uses:** if you **normalise** every vector to length 1 first, the
denominator becomes 1 and cosine similarity is just the dot product — a single
multiply-and-add, which hardware does extremely fast.

```python
self.emb = self.model.encode(..., normalize_embeddings=True)
...
scores = self.emb @ q    # one matrix multiply against all 5,183 documents
```

That `@` is the entire search. 5,183 × 384 multiply-adds, about 20 ms.

## The silent bug almost everyone ships

```python
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
```

The BGE family of models was **trained** with queries carrying this instruction prefix
and documents carrying none. The model learned that queries and documents are
different kinds of thing.

If you skip the prefix, nothing errors. Nothing warns you. Your results are just
quietly a few points worse, forever. This class of bug — a model used slightly
differently than it was trained — is extremely common and invisible without evaluation.

**Lesson: read the model card.** Every embedding model has its own conventions
(prefixes, pooling, max length, normalisation). They are not interchangeable.

## Why dense retrieval fails

It's genuinely better than BM25 on this project's data — 0.7077 vs 0.6653 nDCG@10 on
SciFact — but it fails in ways BM25 doesn't:

**It blurs exact tokens.** Product codes, form numbers, error codes, rare names get
smoothed into "something in this general area." If a user searches `EPF Form 19`, a
dense retriever may happily return Form 13 — semantically almost identical, completely
useless.

**It fails confidently.** BM25 returning nothing is obvious: no words matched. A dense
retriever *always* returns its 10 nearest neighbours, even when the nearest is
irrelevant, with a similarity score that looks reassuringly high.

**It can't exceed its training domain.** An embedding model that never saw legal or
medical text during training doesn't understand what makes two legal documents similar.

Notice these are precisely BM25's strengths. **That complementarity is the hybrid
argument** — and [doc 4](04-fusion-combining-retrievers.md) tests whether it holds up.

## Bi-encoder vs cross-encoder

This is the single most important architectural distinction in retrieval.

What `dense.py` implements is a **bi-encoder**:

```
embed(query)     ─┐
                  ├─→  cosine similarity  →  score
embed(document)  ─┘

The two texts are encoded SEPARATELY and never meet inside the model.
```

Because documents are encoded independently of the query, you can embed your whole
corpus once, at startup, and reuse it for every query forever. **That's what makes it
fast enough to search millions of documents.**

It's also the limitation: the model producing the document vector had no idea what
would be asked. It had to compress the document into 384 numbers covering every
possible future question. Fine detail gets lost.

The alternative — feeding query and document in *together* — is a **cross-encoder**,
and that's [doc 5](05-reranking-cross-encoders.md).

→ Next: [Fusion: combining retrievers](04-fusion-combining-retrievers.md)
