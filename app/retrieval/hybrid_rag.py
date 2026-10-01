"""
Pipeline 2 -- Hybrid retrieval: BM25 (lexical) + vector (semantic), merged
with Reciprocal Rank Fusion. Same RRF pattern used in the earlier
retrieval-service project -- catches both exact-term matches ("vitamin D")
and semantic paraphrases ("sun exposure nutrient") that pure BM25 would miss.
"""
from app.ingest import get_bm25_index, get_chroma_collection


def _bm25_search(query: str, top_k: int) -> list[dict]:
    bm25, docs = get_bm25_index()
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(scores, docs), key=lambda p: p[0], reverse=True)
    return [
        {"text": doc["text"], "source": doc["id"], "score": float(score)}
        for score, doc in ranked[:top_k]
        if score > 0
    ]


def _vector_search(query: str, top_k: int) -> list[dict]:
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    return [
        {"text": text, "source": doc_id, "score": 1.0 - dist}
        for text, doc_id, dist in zip(
            results["documents"][0], results["ids"][0], results["distances"][0]
        )
    ]


def retrieve(query: str, top_k: int = 3, k_rrf: int = 60) -> list[dict]:
    bm25_results = _bm25_search(query, top_k * 3)
    vector_results = _vector_search(query, top_k * 3)

    fused_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}
    for ranked_list in (bm25_results, vector_results):
        for rank, item in enumerate(ranked_list, start=1):
            fused_scores[item["source"]] = fused_scores.get(item["source"], 0.0) + 1.0 / (
                k_rrf + rank
            )
            doc_lookup[item["source"]] = item

    merged = sorted(fused_scores.items(), key=lambda p: p[1], reverse=True)
    return [{**doc_lookup[src], "score": score} for src, score in merged[:top_k]]
