import os
import time
from src.dataset import load_beir
from src.baseline import BM25Retriever, TfidfRetriever
from src.embedding import EmbeddingRetriever
from src.fusion import HybridRetriever, RerankRetriever
from src.evaluate import evaluate

K = 10

print("Loading SciFact...")
chunks, queries, qrels = load_beir("data/scifact")
print(f"{len(chunks)} chunks, {len(queries)} labeled queries\n")

bm25 = BM25Retriever(chunks)
dense = EmbeddingRetriever(chunks, index_path="data/scifact.faiss")
hybrid = HybridRetriever([bm25, dense])
systems = {
    "TF-IDF": TfidfRetriever(chunks),
    "BM25": bm25,
    "Dense (MiniLM)": dense,
    "Hybrid (RRF)": hybrid,
    "Hybrid + rerank": RerankRetriever(hybrid),
}
if os.path.isdir("models/minilm-scifact"):  # produced by finetune.py
    tuned = EmbeddingRetriever(chunks, "models/minilm-scifact", index_path="data/scifact-ft.faiss")
    systems["Dense, fine-tuned on SciFact train"] = tuned
    systems["Hybrid: BM25 + fine-tuned (RRF)"] = HybridRetriever([bm25, tuned])
if os.environ.get("DATABASE_URL"):  # set it after `docker compose up -d && python load_pg.py`
    from src.pgstore import PgVectorRetriever
    systems["Dense via pgvector (HNSW)"] = PgVectorRetriever(dense.model)

results = {}
for name, retriever in systems.items():
    print(f"Evaluating {name}...")
    t0 = time.time()
    metrics = evaluate(retriever, queries, qrels, k=K)
    metrics["ms/query"] = round(1000 * (time.time() - t0) / len(queries), 1)
    results[name] = metrics

cols = list(next(iter(results.values())).keys())
print("\n| System | " + " | ".join(cols) + " |")
print("|---" * (len(cols) + 1) + "|")
for name, m in results.items():
    print(f"| {name} | " + " | ".join(f"{m[c]:.4f}" if c != "ms/query" else str(m[c]) for c in cols) + " |")
