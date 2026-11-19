"""Loading BEIR-format data.

BEIR is a standard collection of retrieval benchmarks. Every dataset in it has
the same three pieces, which is the shape you'll see in almost all retrieval work:

  corpus  : the documents you search over      {doc_id -> {title, text}}
  queries : the questions people ask           {query_id -> question text}
  qrels   : "query relevance judgements" -- a human-labelled answer key saying
            which doc_ids are correct for each query_id. This is what lets you
            measure accuracy instead of eyeballing results.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def _jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_corpus():
    """doc_id -> {"title":..., "text":..., "full":...}

    `full` is title and body glued together. Titles carry a lot of signal in
    retrieval, so we search over both rather than the body alone.
    """
    out = {}
    for r in _jsonl(DATA / "corpus.jsonl"):
        title, text = (r.get("title") or "").strip(), (r.get("text") or "").strip()
        out[str(r["_id"])] = {
            "title": title,
            "text": text,
            "full": (title + ". " + text).strip(". ").strip(),
        }
    return out


def load_queries():
    return {str(r["_id"]): (r.get("text") or "").strip()
            for r in _jsonl(DATA / "queries.jsonl")}


def load_qrels():
    """query_id -> {doc_id: relevance_score}

    relevance_score > 0 means "this document is a correct answer". Some datasets
    grade relevance 0/1/2; SciFact is mostly 1. The metrics handle both.
    """
    out = {}
    for r in _jsonl(DATA / "qrels.jsonl"):
        qid, did, s = str(r["query-id"]), str(r["corpus-id"]), int(r["score"])
        if s > 0:
            out.setdefault(qid, {})[did] = s
    return out


def load_all():
    corpus, queries, qrels = load_corpus(), load_queries(), load_qrels()
    # Only keep queries that actually have an answer key -- you can't score the rest.
    queries = {q: t for q, t in queries.items() if q in qrels}
    return corpus, queries, qrels
