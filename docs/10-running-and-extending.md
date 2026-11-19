# 10. Running and extending

## Setup

```bash
git clone https://github.com/vvvvvivekkk/hybrid-retrieval-lab
cd hybrid-retrieval-lab
pip install -r requirements.txt
```

Models download automatically from HuggingFace on first use (~500 MB total) and cache
in `~/.cache/huggingface/`.

## The full run

```bash
python download_data.py --dataset scifact
python cache_runs.py     --dataset scifact --device cuda
python tune.py           --dataset scifact
python run_ablation.py   --dataset scifact --device cuda
python compare_rerankers.py  --dataset scifact --device cuda
python benchmark_ann.py      --dataset scifact --device cuda
python benchmark_metadata.py --dataset scifact --device cuda
```

Drop `--device cuda` (or pass `cpu`) without a GPU.

### Timings

| step | RTX 4060 | CPU |
|---|---|---|
| download | ~20 s | ~20 s |
| cache_runs | ~30 s | ~10 min |
| tune | instant | instant |
| compare_rerankers | ~1 min | ~25 min |
| benchmark_ann | ~40 s | ~10 min |

**The cross-encoders are the only part that truly needs a GPU** — `bge-reranker-base`
runs ~21 s/query on CPU versus well under a second on a 4060. Everything else is
tolerable either way.

## Adding a dataset

Every script takes `--dataset`, so any BEIR benchmark works unchanged:

```bash
python download_data.py --dataset fiqa          # financial QA, 57k docs
python cache_runs.py    --dataset fiqa --device cuda
python tune.py          --dataset fiqa
```

Worth trying, and roughly in this order:

| dataset | docs | why it's interesting |
|---|---|---|
| `nfcorpus` | 3.6k | real user search queries — already in this repo |
| `fiqa` | 57k | financial QA, closest to MS MARCO, where MiniLM should shine |
| `arguana` | 8.7k | argument retrieval — the query *is* a document |
| `trec-covid` | 171k | big enough that an ANN index starts to matter |
| `scidocs` | 25k | citation prediction |

**`fiqa` is the highest-value next run.** It's much more MS-MARCO-like than either
dataset here, so it's the cleanest test of whether the reranker inversion in
[doc 5](05-reranking-cross-encoders.md) holds up.

## Swapping the embedding model

In `src/dense.py`, or via `run_ablation.py --embed-model`:

| model | dims | notes |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | current default; fast, strong |
| `BAAI/bge-base-en-v1.5` | 768 | better, ~3× slower |
| `intfloat/e5-base-v2` | 768 | needs `"query: "` / `"passage: "` prefixes |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | very fast, no prefix, weaker |

**Check the prefix convention every time.** E5 wants `query:` and `passage:`; BGE
wants its instruction on queries only; MiniLM wants nothing. Getting this wrong costs
accuracy with no error message — see [doc 3](03-embeddings-meaning-search.md).

The README notes this honestly: swapping the embedding model would likely move the
numbers more than any fusion tuning did. That's the highest-leverage experiment left.

## Adapting it to your own documents

The realistic path from this repo to a working system:

**1. Get your documents into the BEIR shape.** Write `corpus.jsonl` with
`{"_id", "title", "text"}` per line. Everything else works unchanged.

**2. Build a small answer key.** This is the step people skip and it is the step that
matters. Collect 50–100 real queries — from logs, from colleagues, from support
tickets — and for each, label which documents are correct. A day of tedious work buys
you the ability to measure every change you make afterwards. Without it you are back
to "looks better to me."

**3. Chunk long documents.** This repo uses whole abstracts, which are short. Real
documents need splitting into passages (typically 200–500 tokens with overlap).
Chunking strategy often matters more than model choice, and this repo doesn't cover it
— a genuine gap in the project.

**4. Run `tune.py` on your split.** Do not assume `k=10, 0.5:1` transfers. It won on
both datasets here, which is encouraging, not conclusive.

**5. Test both rerankers on your data, and be prepared to use neither.**

## Debugging

**Recall@50 is low (<0.8).** Retrieval is the problem, not ranking. Check chunking,
check the query prefix, check tokenisation, try a stronger embedding model. Do not add
a reranker — it cannot fix this.

**Recall is high, nDCG is low.** The documents are there, badly ordered. Tune fusion,
try a reranker. This is the case where reranking can genuinely help.

**The reranker made things worse.** Expected, roughly half the time — see
[doc 5](05-reranking-cross-encoders.md). Try the other model, then try none.

**Out of memory on GPU.** Lower `batch_size` in `DenseRetriever` or `Reranker`. An
8 GB card handles batch 64 for bge-small and 32 for cross-encoders comfortably.

**Results differ slightly between runs.** Embeddings are deterministic; HNSW
construction is not (it depends on insertion order). nDCG differences under ~0.005
between runs are not meaningful.

**A metric is suspiciously perfect (1.0) or zero.** Almost always a key-type mismatch
— qrels IDs are strings, your run IDs are ints, and nothing intersects. Print one
entry from each and compare types. This repo forces `str()` everywhere in `load.py`
for exactly this reason.

→ Next: [What this teaches you](11-what-this-teaches-you.md)
