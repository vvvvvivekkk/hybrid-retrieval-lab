"""BM25 -- the keyword searcher.

BM25 scores a document for a query by asking, for each query word:
  1. How often does this word appear in the document? (more = better, but with
     diminishing returns -- the 10th occurrence adds far less than the 2nd)
  2. How rare is this word across the whole corpus? (rarer = more informative.
     "the" tells you nothing; "hyperkalemia" tells you a lot)
  3. How long is the document? (long docs get penalised, so they don't win just
     by containing more words)

It has no idea what words *mean*. "car" and "automobile" are unrelated to it.
That blindness is exactly why we pair it with a dense retriever.
"""
import re
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+")

# Very common words carry no discriminating signal and just add noise.
STOP = set("""a an the and or but if then than that this these those of in on at to for
from by with without as is are was were be been being it its do does did not no so such
we you they he she i there here what which who whom when where how all any both each""".split())


def tokenize(text: str):
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


class SparseRetriever:
    def __init__(self, corpus: dict):
        self.doc_ids = list(corpus.keys())
        self.bm25 = BM25Okapi([tokenize(corpus[d]["full"]) for d in self.doc_ids])

    def search(self, query: str, top_k: int = 100):
        """Returns [(doc_id, score), ...] sorted best-first."""
        scores = self.bm25.get_scores(tokenize(query))
        # argpartition grabs the top_k without fully sorting all 5,183 scores.
        import numpy as np
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[(-scores[idx]).argsort()]
        return [(self.doc_ids[i], float(scores[i])) for i in idx]
