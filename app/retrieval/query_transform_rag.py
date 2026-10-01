"""
Pipeline 3 -- Query transformation: an LLM rewrites the user's often-vague
question into a query that matches how the knowledge base is actually
phrased, before retrieval runs. E.g. "I want to lose weight fast" ->
"caloric deficit strategies for sustainable weight loss" retrieves better
than the original because it uses the corpus's own vocabulary.
"""
from app.config import get_llm
from app.retrieval.hybrid_rag import retrieve as hybrid_retrieve

_REWRITE_SYSTEM = """Rewrite the user's question into a short, specific search
query (under 15 words) using precise nutrition/health terminology, optimized
for retrieving relevant passages from a nutrition knowledge base. Respond
with ONLY the rewritten query, no explanation."""


def transform_query(query: str) -> str:
    llm = get_llm()
    response = llm.invoke([("system", _REWRITE_SYSTEM), ("user", query)])
    return response.content.strip()


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    rewritten = transform_query(query)
    results = hybrid_retrieve(rewritten, top_k=top_k)
    for r in results:
        r["rewritten_query"] = rewritten
    return results
