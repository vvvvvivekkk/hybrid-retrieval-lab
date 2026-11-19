# 1. Why retrieval exists

## The problem

Language models have two hard limits that create this entire field.

**They don't know your documents.** A model is trained on public text up to a cutoff
date. Your company wiki, your PDFs, your support tickets — it has never seen them.

**They can't read everything at once.** A model has a **context window**: the maximum
amount of text it can consider in one go. Even a large window (say 200,000 tokens, very
roughly 150,000 words) is nothing next to a corpus of 50,000 documents. And stuffing
the window is both expensive and counterproductive — models get measurably worse at
finding a fact buried in a huge pile of mostly-irrelevant text.

## The solution, and where it goes wrong

**RAG** — Retrieval-Augmented Generation — is the standard answer:

```
user question
     ↓
[RETRIEVER]  ← searches your documents, returns the ~5 most relevant passages
     ↓
[LLM]        ← reads only those 5 passages, writes the answer
     ↓
answer
```

Here is the part most people skip past: **the retriever is the ceiling on the whole
system.** If it returns the wrong five passages, the model has no path to a correct
answer. It will write something fluent and confident from the wrong source, which is
worse than saying nothing, because now you have a plausible wrong answer.

Teams spend enormous effort on prompts and model choice. The retriever usually
deserves more of that attention than it gets.

## What "relevant" actually means

This sounds obvious and isn't. Consider a query and three documents:

> **Query:** "0-dimensional biomaterials show inductive properties."

- Doc A: a paper about 0-dimensional biomaterials that **confirms** they're inductive
- Doc B: a paper about 0-dimensional biomaterials that **refutes** this
- Doc C: a paper about 3-dimensional biomaterials, inductive properties confirmed

Which are relevant? In this project's SciFact dataset, **A and B both are** — the task
is verifying a claim, so a paper that refutes it is exactly as useful as one that
confirms it. Doc C is not relevant despite sharing most of the vocabulary.

Now the same question for a different dataset:

> **Query:** "does drinking milk cause acne"

Here relevance means "this document helps answer the question" in the ordinary sense.

**These are different tasks wearing the same clothes.** A retrieval component tuned
for one can actively hurt on the other — which is exactly what this project measured
with rerankers (see [doc 5](05-reranking-cross-encoders.md)).

## The two families of retriever

Everything in retrieval is some combination of two ideas:

**Lexical (sparse) — match the words.** BM25 is the standard. Fast, interpretable,
exact. Cannot handle synonyms.

**Semantic (dense) — match the meaning.** Embedding models are the standard. Handles
paraphrase, cross-language, conceptual similarity. Blurs exact tokens, and fails
confidently.

Their failure modes are near-opposites, which is the entire argument for **hybrid**
retrieval: run both, merge the results. Whether that argument survives contact with
measurement is what [doc 4](04-fusion-combining-retrievers.md) and the results tables
are about.

→ Next: [BM25: keyword search](02-bm25-keyword-search.md)
