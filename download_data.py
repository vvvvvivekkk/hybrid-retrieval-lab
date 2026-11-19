"""Download any BEIR dataset. Train/dev splits matter -- see tune.py."""
import argparse, os
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="scifact", help="scifact | nfcorpus | fiqa | trec-covid ...")
a = ap.parse_args()
d = a.dataset
out = f"data/{d}"; os.makedirs(out, exist_ok=True)

load_dataset(f"BeIR/{d}", "corpus",  split="corpus").to_json(f"{out}/corpus.jsonl")
load_dataset(f"BeIR/{d}", "queries", split="queries").to_json(f"{out}/queries.jsonl")

qr = load_dataset(f"BeIR/{d}-qrels")
print("available splits:", list(qr.keys()))
for split in qr:
    qr[split].to_json(f"{out}/qrels_{split}.jsonl")
print("done ->", out)
