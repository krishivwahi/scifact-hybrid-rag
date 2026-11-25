import json
from pathlib import Path
from .ingest import load_corpus

def load_beir(data_dir: str, split: str = "test"):
    data_dir = Path(data_dir)
    docs = {}
    with open(data_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[d["_id"]] = (d.get("title", "") + ". " + d["text"]).strip(". ")
    chunks = load_corpus(docs)

    all_queries = {}
    with open(data_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            all_queries[d["_id"]] = d["text"]
            
    qrels = {}
    with open(data_dir / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        next(f)  # skip dat header row
        for line in f:
            qid, did, score = line.strip().split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, set()).add(did)
                
    queries = {qid: all_queries[qid] for qid in qrels}
    return chunks, queries, qrels
