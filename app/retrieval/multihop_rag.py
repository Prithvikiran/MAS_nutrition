"""
Pipeline 4 -- Multi-hop decomposition: for questions that really need
multiple independent lookups ("What should a diabetic marathon runner eat
before and after a race?" touches diabetes AND athletic performance AND
recovery nutrition), a single retrieval call tends to only surface docs for
whichever sub-topic embeds "loudest." This pipeline splits the question into
sub-questions, retrieves for each independently, and merges/deduplicates.
"""
import json

from app.config import get_llm
from app.retrieval.hybrid_rag import retrieve as hybrid_retrieve

_DECOMPOSE_SYSTEM = """Break the user's question into 1-3 independent
sub-questions that, together, cover everything needed to answer it. If the
question is already simple and single-topic, return just the one question
unchanged. Respond with ONLY a JSON array of strings, no other text."""


def decompose_query(query: str) -> list[str]:
    llm = get_llm()
    response = llm.invoke([("system", _DECOMPOSE_SYSTEM), ("user", query)])
    try:
        sub_questions = json.loads(response.content.strip())
        if isinstance(sub_questions, list) and sub_questions:
            return sub_questions
    except json.JSONDecodeError:
        pass
    return [query]  # fall back to the original question if parsing fails


def retrieve(query: str, top_k_per_hop: int = 2) -> list[dict]:
    sub_questions = decompose_query(query)

    seen_sources = set()
    merged: list[dict] = []
    for sub_q in sub_questions:
        for result in hybrid_retrieve(sub_q, top_k=top_k_per_hop):
            if result["source"] not in seen_sources:
                seen_sources.add(result["source"])
                result["sub_question"] = sub_q
                merged.append(result)

    return merged
