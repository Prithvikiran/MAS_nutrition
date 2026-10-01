"""
Pipeline 1 -- Naive RAG: embed the raw query, take top-k by vector
similarity, done. No query rewriting, no lexical search, no reranking.
This is the baseline every other pipeline in this repo is measured against.
"""
from app.ingest import get_chroma_collection


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    return [
        {"text": text, "source": doc_id, "score": 1.0 - dist}
        for text, doc_id, dist in zip(
            results["documents"][0], results["ids"][0], results["distances"][0]
        )
    ]
