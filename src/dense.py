"""Dense vector search -- the meaning-based searcher.

An *embedding model* turns a piece of text into a fixed-length list of numbers
(here, 384 of them) chosen so that texts with similar meaning end up close
together in that 384-dimensional space. "Close" is measured with cosine
similarity: the angle between two vectors, 1.0 = identical direction.

So retrieval becomes: embed every document once up front, embed the query at
search time, and find the document vectors nearest the query vector.

Why it beats BM25 sometimes: it matches paraphrases with no shared words.
Why it loses sometimes: it blurs exact tokens. Rare identifiers, form numbers
and proper nouns get smoothed into "something sciency", and it retrieves
something plausible-but-wrong with great confidence.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

# BGE models were trained expecting queries to carry this instruction prefix,
# and documents to carry none. Skipping it costs a few accuracy points -- a
# very common silent bug in homemade RAG stacks.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    def __init__(self, corpus: dict, model_name="BAAI/bge-small-en-v1.5",
                 device=None, batch_size=64):
        self.model = SentenceTransformer(model_name, device=device)
        self.doc_ids = list(corpus.keys())
        self.emb = self.model.encode(
            [corpus[d]["full"] for d in self.doc_ids],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # makes cosine similarity a plain dot product
            show_progress_bar=True,
        ).astype(np.float32)

    def search(self, query: str, top_k: int = 100):
        q = self.model.encode([QUERY_PREFIX + query], convert_to_numpy=True,
                              normalize_embeddings=True).astype(np.float32)[0]
        scores = self.emb @ q           # one dot product against every document
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[(-scores[idx]).argsort()]
        return [(self.doc_ids[i], float(scores[i])) for i in idx]
