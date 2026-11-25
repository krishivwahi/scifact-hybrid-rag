FROM python:3.14-slim
WORKDIR /app

COPY requirements.txt .
# CPU-only torch: the default CUDA wheels add several GB
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ src/
COPY app.py ./
# Bake the corpus, the fine-tuned model and its prebuilt FAISS index (run download_data.py + finetune.py first)
# into the image: builds don't depend on the dataset host, and cold starts never re-embed
COPY data/scifact/ data/scifact/
COPY data/scifact-ft.faiss data/
COPY models/minilm-scifact/ models/minilm-scifact/
RUN python -c "from src.pipeline import build_retriever; build_retriever()"
ENV HF_HUB_OFFLINE=1 PORT=8080

# LLM_API_KEY comes from Secret Manager at deploy time, never baked in
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port $PORT"]
