"""
Ingestion: builds (or reuses) the ChromaDB collection and an in-memory BM25
index from data/sample_nutrition_docs.json. All four retrieval pipelines in
app/retrieval/ pull from these two shared indices, so comparisons between
pipelines are apples-to-apples (same corpus, same chunking -- these docs are
already short enough to use as single chunks).
"""
import chromadb
from rank_bm25 import BM25Okapi

from app.config import CHROMA_DIR, load_docs

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection(name="nutrition_docs")

_docs = load_docs()
_bm25_tokens = [doc["text"].lower().split() for doc in _docs]
_bm25_index = BM25Okapi(_bm25_tokens)


def get_chroma_collection():
    if _collection.count() == 0:
        _collection.add(
            ids=[d["id"] for d in _docs],
            documents=[d["text"] for d in _docs],
            metadatas=[{"topic": d["topic"]} for d in _docs],
        )
    return _collection


def get_bm25_index():
    return _bm25_index, _docs
