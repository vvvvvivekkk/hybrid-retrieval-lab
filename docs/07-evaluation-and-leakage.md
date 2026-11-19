# 7. Evaluation and leakage

**File in this repo:** `tune.py`

This chapter is about a mistake this project actually made, caught, and fixed. It is
the most transferable thing here, because it applies to every ML project you will ever
work on.

## What happened

The first version swept the RRF constant `k` across every value, scored each on the
**test set**, and reported the best:

> "At k=10 nDCG@10 = 0.7188, which beats dense alone at 0.7077. Hybrid wins."

That number was not wrong arithmetic. It was **measuring the wrong thing.**

## Why it's wrong

The test set exists to estimate performance on queries the system has **never seen**.
The moment you use it to *choose* between configs, it stops being unseen — you have
fitted a decision to it.

Try 24 configurations, pick the best-scoring one, and some of that winning margin is
real signal and some is **that config happening to suit the noise in those particular
300 queries**. You can't tell which is which. The reported number is optimistic by an
unknown amount, and the gap only reveals itself when you deploy and performance is
worse than your slide said.

This is **test-set leakage** (or "training on the test set"). Public leaderboards are
littered with it.

## The fix

```
1. Grid-search every config on TRAIN (or validation).
2. Pick the single best by TRAIN score. Freeze it.
3. Run that one frozen config on TEST. Report that number only.
```

The test set is touched **once**, after every decision is locked. That's the whole
protocol.

```python
tuned = [(score(tune_split, k, w)["nDCG@10"], k, w) for k, w in GRID]
tuned.sort(reverse=True)
best_train, bk, bw = tuned[0]       # chosen WITHOUT looking at test
honest = score("test", bk, bw)      # test touched once
```

## What it cost

`tune.py` also computes the leaky number, so the inflation is visible:

| | SciFact | NFCorpus |
|---|---|---|
| best-on-test (leaked) | 0.7221 | 0.3562 |
| tuned on train/validation (honest) | **0.7069** | **0.3562** |
| **inflation** | **+2.16%** | **+0.00%** |

Two things worth absorbing.

**First: the project's headline claim changed.** Honestly tuned, SciFact hybrid scores
0.7069 against dense alone at 0.7077. **It ties. It does not win.** Hybrid earns its
place on that dataset only through recall (0.9517 vs 0.9250). The original "hybrid
wins" conclusion was an artifact of leakage.

2.2% is roughly the margin between adjacent entries on a public leaderboard.

**Second, and more subtle: NFCorpus showed zero inflation.** Validation and test picked
the same config, so leakage cost nothing there.

**Leakage is not a fixed tax — it's a variance effect.** It's large when your
evaluation set is small, narrow, or unrepresentative, and invisible when it's
well-behaved. You cannot predict which you have. You can only follow the protocol
every time.

## Splits

| split | purpose | how often you may look |
|---|---|---|
| **train** | fit parameters, or (here) search hyperparameters | freely |
| **validation / dev** | choose between configs | freely |
| **test** | final unbiased estimate | **once, at the end** |

This project has no learned parameters — nothing is trained. **Hyperparameter search
is still fitting.** Any decision informed by data is fitting to that data, whether
gradient descent or a `for` loop over `k` values. The protocol applies either way, and
that's the part people miss.

```python
tune_split = a.tune_split or ("validation" if "validation" in splits else "train")
```

Prefer validation when it exists; fall back to train.

## Leakage in other forms

You will meet this bug wearing other costumes:

- **Temporal leakage.** Predicting stock prices using tomorrow's data for today's
  features. Any time-ordered data needs a time-ordered split, never a random one.
- **Duplicate leakage.** The same document in train and test under different IDs. Your
  model memorised rather than generalised.
- **Preprocessing leakage.** Computing normalisation statistics (mean, IDF, scaler)
  over the full dataset *before* splitting. The training set now carries information
  about the test set.
- **Repeated peeking.** No single run looks at test, but you ran 40 experiments and
  kept the one with the best test score. Same problem, slower.

The last one is the hardest to avoid honestly, because it doesn't feel like cheating.

## Remaining honest caveats in this repo

Documented in the README, and worth stating plainly:

- **Reranker comparisons use 80 queries**, enough to show a sign flip, not enough for
  a tight confidence interval.
- **No significance testing.** Differences under ~0.01 nDCG — MiniLM's +0.0089 on
  NFCorpus — should be treated as noise until bootstrapped.
- **Two datasets**, both smallish and scientific-ish.

Naming your own weaknesses isn't modesty, it's credibility. A reviewer who finds a
limitation you didn't mention stops trusting everything else you wrote.

→ Next: [Scaling: indexes and filters](08-scaling-indexes-and-filters.md)
