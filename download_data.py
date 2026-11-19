from datasets import load_dataset
import os
os.makedirs("data", exist_ok=True)
load_dataset("BeIR/scifact", "corpus",  split="corpus").to_json("data/corpus.jsonl")
load_dataset("BeIR/scifact", "queries", split="queries").to_json("data/queries.jsonl")
load_dataset("BeIR/scifact-qrels",      split="test").to_json("data/qrels.jsonl")
print("done")
