import os

from .baseline import BM25Retriever
from .dataset import load_beir
from .embedding import EmbeddingRetriever
from .fusion import HybridRetriever

TUNED = "models/minilm-scifact"  # produced by finetune.py


def build_retriever(data_dir: str = "data/scifact"):
    """Best configuration from the ablation: BM25 + fine-tuned dense, fused with RRF.
    Falls back to the base model if finetune.py hasn't been run. The cross-encoder reranker is left
    out: it added no nDCG on SciFact and costs far more latency."""
    chunks, _queries, _qrels = load_beir(data_dir)
    if os.path.isdir(TUNED):
        dense = EmbeddingRetriever(chunks, TUNED, index_path=f"{data_dir}-ft.faiss")
    else:
        dense = EmbeddingRetriever(chunks, index_path=f"{data_dir}.faiss")
    return HybridRetriever([BM25Retriever(chunks), dense])
