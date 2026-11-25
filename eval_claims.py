"""Claim verification on SciFact dev: single-shot RAG vs the ADK multi-agent checker.

Gold labels come from Postgres (run load_pg.py first). Usage: python eval_claims.py [n_claims]
"""
import asyncio
import json
import re
import sys
import time

from sklearn.metrics import accuracy_score, f1_score

from src.agent import Verdict, build_agent, check_claim
from src.pgstore import connect
from src.pipeline import build_retriever
from src.rag import generate
from src.telemetry import setup_tracing

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
LABELS = ["SUPPORT", "CONTRADICT", "NEI"]

GOLD_SQL = """
SELECT c.claim_id, c.claim, coalesce(max(e.label), 'NEI'), array_remove(array_agg(e.doc_id), NULL)
FROM claims c LEFT JOIN claim_evidence e USING (claim_id)
WHERE c.split = 'dev'
GROUP BY c.claim_id, c.claim
ORDER BY md5(c.claim_id::text)  -- fixed pseudo-random sample
LIMIT %s
"""


def rag_verdict(retriever, claim: str) -> tuple[dict, dict]:
    hits = retriever.search(claim, k=5)
    context = "\n\n".join(f"doc_id {c.doc_id}: {c.text}" for c, _s in hits)
    prompt = (
        f"Context:\n{context}\n\nClaim: {claim}\n\n"
        "Using ONLY the context, label the claim SUPPORT, CONTRADICT or NEI (not enough info). "
        'Reply with JSON only: {"label": ..., "doc_ids": [ids your label relies on], "rationale": "..."}'
    )
    text, stats = generate(prompt)
    match = re.search(r"\{.*\}", text, re.S)
    try:
        return Verdict.model_validate_json(match.group(0)).model_dump(), {**stats, "llm_calls": 1}
    except Exception:  # scored as NEI, but counted so it can't hide in the results
        print(f"  unparseable output: {text[:120]!r}")
        return {"label": "NEI", "doc_ids": []}, {**stats, "llm_calls": 1, "parse_error": 1}


def with_retry(fn, *args):
    """Returns None if every attempt fails, so one overloaded claim doesn't lose the whole run."""
    for attempt in range(3):
        try:
            return fn(*args)
        except Exception as e:  # free tiers return 429/503 under load; back off and retry
            print(f"  retry after error: {' '.join(str(e).split())[:120]}")
            time.sleep(30 * (attempt + 1))
    return None


def score(gold, preds, stats):
    y_true = [g[2] for g in gold]
    y_pred = [p["label"] for p in preds]
    cited = [(set(map(str, p["doc_ids"])), set(g[3])) for g, p in zip(gold, preds) if p["label"] != "NEI" and p["doc_ids"]]
    n = len(stats)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro-F1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "citation precision": sum(len(c & g) / len(c) for c, g in cited) / len(cited) if cited else 0.0,
        "LLM calls/claim": sum(s["llm_calls"] for s in stats) / n,
        "tokens/claim": sum(s["tokens_in"] + s["tokens_out"] for s in stats) / n,
        "latency s/claim": sum(s["latency_s"] for s in stats) / n,
        "$/claim": sum(s["cost_usd"] for s in stats) / n,
        "parse errors": sum(s.get("parse_error", 0) for s in stats),
    }


def main():
    setup_tracing()
    with connect() as conn:
        gold = conn.execute(GOLD_SQL, (N,)).fetchall()
    retriever = build_retriever()
    runner = build_agent(retriever)

    outputs = {}
    for name, run in [
        ("Single-shot RAG", lambda claim: rag_verdict(retriever, claim)),
        ("ADK multi-agent", lambda claim: asyncio.run(check_claim(runner, claim))),
    ]:
        outputs[name] = []
        for i, (_cid, claim, label, _docs) in enumerate(gold, 1):
            out = with_retry(run, claim)
            outputs[name].append(out)
            print(f"{name} {i}/{len(gold)}: gold={label} pred={out[0]['label'] if out else 'FAILED'}")

    # score both systems on the same claims: drop any claim either system failed on
    keep = [i for i in range(len(gold)) if all(outs[i] for outs in outputs.values())]
    results = {name: score([gold[i] for i in keep], [outs[i][0] for i in keep], [outs[i][1] for i in keep])
               for name, outs in outputs.items()}

    with open("results_claims.json", "w") as f:
        json.dump({"n_claims": len(keep), "skipped_after_retries": len(gold) - len(keep), "results": results}, f, indent=2)
    cols = list(next(iter(results.values())))
    print(f"\n{len(keep)} SciFact dev claims ({len(gold) - len(keep)} skipped after retries)")
    print("| System | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, m in results.items():
        print(f"| {name} | " + " | ".join(
            f"{v:.6f}" if k == "$/claim" else str(v) if k == "parse errors" else f"{v:.3f}" for k, v in m.items()) + " |")


if __name__ == "__main__":
    main()
