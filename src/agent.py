"""Multi-agent claim checker (Google ADK).

claim_checker (SequentialAgent)          hierarchical delegation
├── researcher (LlmAgent + search tool)  ReAct: reason -> search -> observe -> search again
└── verify_loop (LoopAgent, max 2 rounds)
    ├── verifier (LlmAgent, JSON verdict)
    └── critic   (LlmAgent + approve tool) self-reflection: approves, or sends a critique back
"""
import os
import time
import uuid
from typing import Literal

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import ToolContext
from google.genai import types
from pydantic import BaseModel, ConfigDict

from .telemetry import cost_usd

load_dotenv()
# shortcut: ADK talks to Gemini natively, so the agent needs a gemini-* LLM_MODEL; wrap with LiteLlm for other providers
os.environ.setdefault("GOOGLE_API_KEY", os.environ.get("LLM_API_KEY", ""))
MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")


class Verdict(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)  # models often emit doc ids as JSON numbers

    label: Literal["SUPPORT", "CONTRADICT", "NEI"]
    doc_ids: list[str]
    rationale: str


def approve(tool_context: ToolContext) -> dict:
    """Call this when the verdict is correct and every cited doc_id really states what the verdict relies on."""
    tool_context.actions.escalate = True  # ends the verify loop
    return {"approved": True}


def build_agent(retriever):
    def search_abstracts(query: str) -> list[dict]:
        """Searches ~5,000 biomedical abstracts and returns the 5 most relevant passages with their doc_id.
        Call it again with a rephrased query if the results are off-topic."""
        return [{"doc_id": c.doc_id, "text": c.text} for c, _s in retriever.search(query, k=5)]

    researcher = LlmAgent(
        name="researcher",
        model=MODEL,
        tools=[search_abstracts],
        output_key="evidence",
        instruction=(
            "You gather evidence about a scientific claim. Think about what finding would confirm it and what "
            "would refute it, then call search_abstracts with 2-3 differently phrased queries. Finish with a list "
            "of the relevant passages: doc_id plus a short verbatim quote each. Do not judge the claim yourself."
        ),
    )
    verifier = LlmAgent(
        name="verifier",
        model=MODEL,
        include_contents="none",
        output_schema=Verdict,
        output_key="verdict",
        instruction=(
            "Claim: {claim}\n\nEvidence gathered:\n{evidence}\n\nCritique of your previous verdict, if any: {critique?}\n\n"
            "Label the claim SUPPORT or CONTRADICT only if a passage directly states it; otherwise NEI "
            "(not enough info). doc_ids: only the documents your label relies on, empty for NEI."
        ),
    )
    critic = LlmAgent(
        name="critic",
        model=MODEL,
        include_contents="none",
        tools=[approve],
        output_key="critique",
        instruction=(
            "Claim: {claim}\n\nEvidence gathered:\n{evidence}\n\nProposed verdict: {verdict}\n\n"
            "Check the verdict strictly: does each cited doc_id's quote actually say what the label needs? "
            "Is a weaker label (NEI) more honest? If the verdict holds, call approve. Otherwise reply with a "
            "short critique saying what to change."
        ),
    )
    root = SequentialAgent(
        name="claim_checker",
        sub_agents=[researcher, LoopAgent(name="verify_loop", sub_agents=[verifier, critic], max_iterations=2)],
    )
    return InMemoryRunner(agent=root, app_name="claim_checker")


async def check_claim(runner: InMemoryRunner, claim: str) -> tuple[dict, dict]:
    user = "user"
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user, session_id=str(uuid.uuid4()), state={"claim": claim})
    t0 = time.perf_counter()
    tokens_in = tokens_out = calls = 0
    async for event in runner.run_async(
        user_id=user, session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=claim)]),
    ):
        if u := event.usage_metadata:
            calls += 1
            tokens_in += u.prompt_token_count or 0
            tokens_out += (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
    session = await runner.session_service.get_session(app_name=runner.app_name, user_id=user, session_id=session.id)
    stats = {"llm_calls": calls, "tokens_in": tokens_in, "tokens_out": tokens_out,
             "latency_s": round(time.perf_counter() - t0, 2), "cost_usd": cost_usd(tokens_in, tokens_out)}
    return session.state.get("verdict") or {"label": "NEI", "doc_ids": [], "rationale": "no verdict"}, stats
