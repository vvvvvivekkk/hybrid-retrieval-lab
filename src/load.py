"""Loading BEIR-format data.

BEIR is a standard collection of retrieval benchmarks. Every dataset has the same
three pieces, the shape you'll meet in nearly all retrieval work:

  corpus  : the documents you search over       {doc_id -> {title, text}}
  queries : the questions people ask            {query_id -> question text}
  qrels   : "query relevance judgements" -- a human-labelled answer key saying
            which doc_ids are correct for each query_id.

The SPLIT matters. qrels come as train / (validation) / test. Anything you tune --
fusion weights, the RRF constant, how many candidates to rerank -- must be tuned on
train or validation and reported on test. Tuning on test is "leakage": your numbers
quietly describe how well you fitted the answer key, not how well the system works
on queries it has never seen.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_corpus(dataset="scifact"):
    """doc_id -> {"title", "text", "full"}. `full` glues title to body: titles carry
    a lot of retrieval signal, so we search over both."""
    out = {}
    for r in _jsonl(DATA / dataset / "corpus.jsonl"):
        title, text = (r.get("title") or "").strip(), (r.get("text") or "").strip()
        out[str(r["_id"])] = {"title": title, "text": text,
                              "full": (title + ". " + text).strip(". ").strip()}
    return out


def load_queries(dataset="scifact"):
    return {str(r["_id"]): (r.get("text") or "").strip()
            for r in _jsonl(DATA / dataset / "queries.jsonl")}


def load_qrels(dataset="scifact", split="test"):
    """query_id -> {doc_id: relevance}. Relevance can be graded (0/1/2); 0 means
    'judged and not relevant', so we drop it."""
    out = {}
    for r in _jsonl(DATA / dataset / f"qrels_{split}.jsonl"):
        s = int(r["score"])
        if s > 0:
            out.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = s
    return out


def available_splits(dataset="scifact"):
    return sorted(p.stem.replace("qrels_", "") for p in (DATA / dataset).glob("qrels_*.jsonl"))


def load_split(dataset="scifact", split="test", max_queries=None):
    """Returns (corpus, queries, qrels) for one split, queries filtered to those
    that actually have an answer key."""
    corpus = load_corpus(dataset)
    qrels = load_qrels(dataset, split)
    queries = {q: t for q, t in load_queries(dataset).items() if q in qrels}
    if max_queries:
        keep = list(queries)[:max_queries]
        queries = {q: queries[q] for q in keep}
        qrels = {q: qrels[q] for q in keep}
    return corpus, queries, qrels


# backwards-compatible alias
def load_all(dataset="scifact"):
    return load_split(dataset, "test")
