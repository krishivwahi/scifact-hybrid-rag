"""Fine-tune all-MiniLM-L6-v2 on SciFact train claims -> evidence abstracts.

Contrastive loss with in-batch negatives: each claim is pulled toward its evidence abstract and pushed
away from every other abstract in the batch. Hyperparameters are fixed up front; 10% of the train
claims are held out as a validation set, and the test set is never touched here (run_eval.py reports it).
"""
import random
from pathlib import Path

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from src.dataset import load_beir
from src.embedding import EmbeddingRetriever
from src.evaluate import evaluate

BASE, OUT = "all-MiniLM-L6-v2", "models/minilm-scifact"

chunks, queries, qrels = load_beir("data/scifact", split="train")
doc_text = {}
for c in chunks:  # first chunk: MiniLM only reads ~256 tokens, and most abstracts are one chunk
    doc_text.setdefault(c.doc_id, c.text)

qids = sorted(queries)
random.Random(0).shuffle(qids)
val_ids = qids[: len(qids) // 10]
pairs = [(queries[q], doc_text[d]) for q in qids[len(val_ids):] for d in qrels[q] if d in doc_text]
print(f"{len(pairs)} training pairs, {len(val_ids)} held-out validation claims")

model = SentenceTransformer(BASE)
args = SentenceTransformerTrainingArguments(
    output_dir="models/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # a claim with 2 evidence docs must not be its own negative
    save_strategy="no",
    logging_steps=10,
    seed=0,
    report_to="none",
)
train = Dataset.from_dict({"anchor": [a for a, _ in pairs], "positive": [p for _, p in pairs]})
SentenceTransformerTrainer(model=model, args=args, train_dataset=train,
                           loss=MultipleNegativesRankingLoss(model)).train()
model.save(OUT)
Path("data/scifact-ft.faiss").unlink(missing_ok=True)  # the cached index holds the previous model's vectors

# sanity check on the held-out train claims only
val_q = {q: queries[q] for q in val_ids}
val_rel = {q: qrels[q] for q in val_ids}
base = EmbeddingRetriever(chunks, BASE, index_path="data/scifact.faiss")
tuned = EmbeddingRetriever(chunks, OUT, index_path="data/scifact-ft.faiss")
print("validation nDCG@10  base:", round(evaluate(base, val_q, val_rel)["nDCG@10"], 4),
      " fine-tuned:", round(evaluate(tuned, val_q, val_rel)["nDCG@10"], 4))
