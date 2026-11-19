# 11. What this teaches you

## The retrieval knowledge

You now know, concretely:

- How BM25 scores documents, and why rare words dominate
- What an embedding is, what cosine similarity measures, why normalisation matters
- The bi-encoder / cross-encoder distinction and the cascade architecture it forces
- Why score fusion needs rank-based methods, and what RRF's `k` controls
- nDCG, Recall@k and MRR — what each measures and when each is the right one
- How ANN indexes trade recall for latency, and when they're premature
- Why post-filtering metadata silently destroys recall

That's genuinely most of applied retrieval. The gaps worth closing next are
**chunking strategy** (how you split long documents) and **query understanding**
(rewriting, expansion, decomposition).

## The transferable lessons — these matter more

**1. Defaults are hyperparameters.** `k=60` is copied from a 2009 paper into every
RAG tutorial. Tuning it moved SciFact nDCG from 0.7040 to 0.7188 — the difference
between hybrid losing and winning. Every magic constant in your stack is an untested
assumption.

**2. Tune on train, report on test — always.** Leakage inflated this project's
headline by 2.2% and reversed its conclusion. It cost *nothing* on the second dataset.
You cannot predict which case you're in, so you follow the protocol every time.

**3. "Bigger model" is not a strategy.** `bge-reranker-base` is 5× larger than MiniLM
and is the worst option on NFCorpus, 6× slower than the model that beats it there.

**4. Measure the pair, not the headline.** "4.6× faster" was true at 69% recall —
useless. Speed without accuracy, or accuracy without latency, is marketing.

**5. Know when not to use the sophisticated tool.** The correct decision about ANN
indexing at 5,000 documents is *don't*. Senior engineers are distinguished by what
they leave out.

**6. Negative results are results.** Three of four "standard" upgrades were neutral or
harmful on at least one dataset. Reporting that is more valuable than a clean win
nobody can reproduce.

**7. Some bugs are invisible to the people testing.** Post-filtering works perfectly
for developers and fails only for users with narrow permissions. Ask who *wouldn't*
notice your bug.

## Interview questions this prepares you for

Real questions, with the answer this project gives you.

**"How would you build search over a company's documents?"**
Hybrid BM25 + dense, RRF fusion, optional reranking — but lead with: *first I'd build
an evaluation set of 50–100 labelled queries, because otherwise we can't tell if any
of it works.* That answer alone separates you from most candidates.

**"When would you use a cross-encoder?"**
Only for reranking a small candidate set, never for first-stage retrieval — nothing
can be precomputed, so it's thousands of times slower. And it's a distribution bet:
on SciFact, MS-MARCO-trained MiniLM *reduced* nDCG by 0.025.

**"Our RAG system returns irrelevant documents. How do you debug it?"**
Measure Recall@k first. High recall + low nDCG is a ranking problem → fusion tuning or
a reranker. Low recall is a retrieval problem → chunking, embedding model, query
handling. A reranker cannot fix low recall; it only reorders what it's given.

**"How do you know your improvement is real?"**
Tuned on train or validation, reported once on test, with the metric and query count
stated. And I'd want a significance test — in this project MiniLM's +0.0089 on
NFCorpus, over 80 queries, is plausibly noise, and I'd say so rather than claim it.

**"What's the difference between nDCG and recall?"**
nDCG cares where correct documents sit; recall only whether they appear. Recall is the
ceiling on everything downstream — if Recall@50 is 0.95, no reranker exceeds 95%.

**"Tell me about a time you were wrong."**
*This project.* I reported hybrid retrieval beating dense search at 0.7188. Then I
noticed I'd tuned the fusion constant on the test set. Tuning honestly on train, the
real number was 0.7069 — a tie, not a win. I corrected the README, kept both numbers
visible, and documented the 2.2% inflation.

That last one is the most valuable thing in this repo. Most candidates answer it with
a disguised humblebrag. A real example of catching your own methodological error,
correcting it publicly, and reporting the number that makes your project look *worse*
is rare and extremely credible.

## What to say about this project in one sentence

> "I built a hybrid retrieval stack and measured every component against labelled
> benchmarks — and found that the textbook RRF constant made it worse than no fusion
> at all, that reranker choice inverts between datasets, and that my own first result
> was inflated 2.2% by test-set leakage."

Nobody follows that with "so you can use LangChain?"

## Where to go next

**Finish this project:** run `--dataset fiqa` to test the reranker inversion on a
third corpus, and add bootstrap significance testing so the small deltas get honest
error bars.

**Then the natural sequel** is evaluating *agents* rather than retrievers — grading
the path an agent takes, not just its final answer, and blocking regressions in CI.
Same discipline, harder problem.

← Back to [the index](README.md)
