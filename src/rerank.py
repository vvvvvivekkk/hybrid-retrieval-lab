"""Cross-encoder reranking -- the accuracy step.

The difference that matters:

  Bi-encoder (what dense.py uses)
      embed(query) and embed(doc) computed SEPARATELY, compared at the end.
      The model never sees the two texts together, so it can't notice that the
      query asks about a *decrease* while the passage reports an *increase*.
      Fast, because document vectors are computed once and reused forever.

  Cross-encoder (this file)
      the query and the document are fed into the model GLUED TOGETHER as one
      input, and attention runs across both. The model can directly weigh every
      query word against every document word. Much more accurate.
      Slow, because nothing can be precomputed -- every (query, doc) pair is a
      fresh forward pass through the model.

That cost is why the architecture is shaped the way it is: cheap retrievers cut
5,183 documents down to 50, then the expensive model reads only those 50. You
get cross-encoder accuracy at roughly 1/100th of cross-encoder cost.
"""
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device=None, batch_size=64):
        self.model = CrossEncoder(model_name, device=device, max_length=512)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates, corpus: dict, top_k: int = 10):
        """candidates: [(doc_id, score), ...] from fusion. Returns re-sorted list."""
        if not candidates:
            return []
        doc_ids = [d for d, _ in candidates]
        pairs = [(query, corpus[d]["full"]) for d in doc_ids]
        scores = self.model.predict(pairs, batch_size=self.batch_size,
                                    show_progress_bar=False)
        ranked = sorted(zip(doc_ids, map(float, scores)), key=lambda x: -x[1])
        return ranked[:top_k]
