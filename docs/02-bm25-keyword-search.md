# 2. BM25 — keyword search

**File in this repo:** `src/sparse.py`

## The intuition

You want to score how well a document matches a query, using only word overlap.
A first attempt: count how many query words appear in the document. This fails
immediately, for three reasons, and BM25 is precisely the fix for all three.

### Problem 1: common words dominate

Query: *"the effect of milk on acne"*. The word "the" appears in every document.
Counting it rewards documents for containing nothing useful.

**Fix — IDF (Inverse Document Frequency).** Weight each word by how rare it is across
the whole corpus. A word in 90% of documents tells you almost nothing; a word in 0.1%
of them is extremely informative.

```
IDF(word) ≈ log( total_documents / documents_containing_word )
```

With 5,183 documents: "the" appears in ~5,100 → `log(5183/5100) ≈ 0.016`, nearly zero.
"hyperkalemia" appears in 3 → `log(5183/3) ≈ 7.5`. That's a 470× weight difference.

### Problem 2: repetition shouldn't scale linearly

A document mentioning "acne" 50 times isn't 50× more relevant than one mentioning it
once. It's maybe 3× more relevant. The 2nd mention tells you a lot; the 40th tells you
nothing new.

**Fix — saturation.** Term frequency is passed through a curve that rises fast then
flattens:

```
         tf × (k1 + 1)
    ──────────────────────
       tf + k1
```

`k1` (typically 1.2–2.0) controls where it flattens. With `k1=1.5`: tf=1 → 1.0,
tf=2 → 1.43, tf=10 → 2.17, tf=50 → 2.42. The curve is nearly flat by tf=10.

### Problem 3: long documents cheat

A 10,000-word document contains more of everything. It would beat a perfectly-focused
200-word document on raw counts.

**Fix — length normalisation.** Divide by document length relative to the corpus
average, with `b` (typically 0.75) controlling how aggressively.

## The whole formula

```
BM25(query, doc) = Σ   IDF(w) ×        tf(w, doc) × (k1 + 1)
                 w∈query          ─────────────────────────────────────
                                  tf(w, doc) + k1 × (1 − b + b × |doc|/avgdl)
```

That's it. Three ideas — rare words matter more, repetition saturates, long documents
get penalised — and it has been the strongest non-neural baseline since 1994.

## Worked example

Corpus of 1,000 documents, average length 100 words. Query: **"milk acne"**.

- "milk" appears in 200 docs → IDF = log(1000/200) = **1.61**
- "acne" appears in 50 docs → IDF = log(1000/50) = **3.00**

**Doc A** (100 words): "milk" ×3, "acne" ×2. Length ratio 1.0.

```
milk: 1.61 × (3 × 2.5)/(3 + 1.5×1.0) = 1.61 × 7.5/4.5 = 2.68
acne: 3.00 × (2 × 2.5)/(2 + 1.5×1.0) = 3.00 × 5.0/3.5 = 4.29
                                                 total = 6.97
```

**Doc B** (400 words): "milk" ×6, "acne" ×2. Length ratio 4.0 → penalty
`(1 − 0.75 + 0.75×4) = 3.25`.

```
milk: 1.61 × (6 × 2.5)/(6 + 1.5×3.25) = 1.61 × 15/10.88 = 2.22
acne: 3.00 × (2 × 2.5)/(2 + 1.5×3.25) = 3.00 × 5/6.88  = 2.18
                                                 total = 4.40
```

Doc B has twice the "milk" mentions and still loses — it's diluted. **This is BM25
doing its job.**

Notice also that "acne" contributed more than "milk" in both, despite appearing less,
purely because it's rarer. That's IDF.

## What BM25 cannot do

It has no concept of meaning. These all score **zero** against "milk acne":

- "dairy consumption and skin breakouts" — perfect semantic match, no shared words
- "lactose and pimples"
- any document in another language

That blindness is the entire reason for the next chapter.

## The code

```python
_TOKEN = re.compile(r"[a-z0-9]+")

def tokenize(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]
```

**Tokenisation** = splitting text into the units you'll match on. This version
lowercases (so "Milk" matches "milk"), keeps only letters and digits, drops
stopwords, and drops single characters.

Stopword removal is partly redundant with IDF — IDF already nearly zeroes "the" — but
it shrinks the index and speeds things up.

```python
self.bm25 = BM25Okapi([tokenize(corpus[d]["full"]) for d in self.doc_ids])
```

Every document is tokenised once at startup. `BM25Okapi` precomputes the IDF table
and document lengths, so queries are fast — **12 ms per query** over 5,183 documents
in this project.

```python
idx = np.argpartition(-scores, k - 1)[:k]
idx = idx[(-scores[idx]).argsort()]
```

A small but real optimisation. Sorting all 5,183 scores to get the top 100 is wasteful.
`argpartition` does a partial sort — it guarantees the top k are in the first k slots,
unordered, in O(n) instead of O(n log n). Then we sort just those k.

→ Next: [Embeddings: meaning search](03-embeddings-meaning-search.md)
