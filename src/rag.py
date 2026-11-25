import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from opentelemetry import trace

from .telemetry import cost_usd

load_dotenv()

_client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL"),
    timeout=60,  # fail fast: overloaded models otherwise hang for minutes behind silent retries
)

_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
_tracer = trace.get_tracer(__name__)

def build_prompt(query: str, retrieved) -> str:
    context = "\n\n".join(
        f"[{i + 1}] {chunk.text}" for i, (chunk, _score) in enumerate(retrieved)
    )
    return (
        "Answer the question using ONLY the context below. "
        "Cite sources by their [number]. "
        "If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

def generate(prompt: str) -> tuple[str, dict]:
    """Streams one completion so time-to-first-token and decode speed can be measured."""
    with _tracer.start_as_current_span("llm.generate") as span:
        t0 = time.perf_counter()
        first = None
        parts, usage = [], None
        stream = _client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        )
        for event in stream:
            if event.usage:
                usage = event.usage
            if event.choices and event.choices[0].delta.content:
                first = first or time.perf_counter()
                parts.append(event.choices[0].delta.content)
        end = time.perf_counter()
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        decode_s = end - (first or end)
        stats = {
            "model": _MODEL,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "ttft_s": round((first or end) - t0, 3),
            "latency_s": round(end - t0, 3),
            "tokens_per_s": round(tokens_out / decode_s, 1) if decode_s > 0 else None,
            "cost_usd": cost_usd(tokens_in, tokens_out),
        }
        span.set_attributes({f"llm.{k}": v for k, v in stats.items() if v is not None})
        return "".join(parts), stats

def answer(query: str, retriever, k: int = 5):
    with _tracer.start_as_current_span("rag.answer") as span:
        span.set_attribute("rag.query", query)
        with _tracer.start_as_current_span("rag.retrieve"):
            retrieved = retriever.search(query, k=k)
        text, stats = generate(build_prompt(query, retrieved))
        return text, retrieved, stats
