from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agent import build_agent, check_claim
from src.pipeline import build_retriever
from src.rag import answer

app = FastAPI(title="SciFact Hybrid RAG", description="Hybrid retrieval + RAG + multi-agent claim checking over SciFact")
retriever = build_retriever()
runner = build_agent(retriever)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=10)


class VerifyRequest(BaseModel):
    claim: str = Field(min_length=3, max_length=500)


@app.get("/health")  # Cloud Run reserves paths ending in "z", so /healthz never reaches the app
def health():
    return {"ok": True}


@app.post("/ask")
def ask(req: AskRequest):
    text, retrieved, stats = answer(req.question, retriever, k=req.k)
    sources = [{"n": i, "doc_id": c.doc_id, "score": s, "text": c.text} for i, (c, s) in enumerate(retrieved, 1)]
    return {"answer": text, "sources": sources, "stats": stats}


@app.post("/verify")
async def verify(req: VerifyRequest):
    verdict, stats = await check_claim(runner, req.claim)
    return {"verdict": verdict, "stats": stats}
