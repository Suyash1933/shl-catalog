"""
Retrieval layer — FAISS semantic search + keyword boosting over SHL catalog.
"""

import json
import re
import threading
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

from catalog import load_catalog, catalog_to_text

INDEX_DIR = Path(__file__).parent / "faiss_index"
INDEX_PATH = INDEX_DIR / "index.faiss"
IDS_PATH = INDEX_DIR / "ids.json"

_lock = threading.Lock()
_model: SentenceTransformer | None = None
_index: faiss.IndexFlatIP | None = None
_catalog: list[dict] | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:  # double-check after acquiring lock
                _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_index():
    """Build FAISS index from catalog and save to disk."""
    catalog = load_catalog()
    model = get_model()

    texts = [catalog_to_text(item) for item in catalog]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(IDS_PATH, "w") as f:
        json.dump(list(range(len(catalog))), f)

    print(f"Built index with {len(catalog)} items, dim={dim}")


def load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load the FAISS index and catalog. Build if missing."""
    global _index, _catalog
    if _index is not None and _catalog is not None:
        return _index, _catalog

    with _lock:
        # Double-check after acquiring lock
        if _index is not None and _catalog is not None:
            return _index, _catalog

        if not INDEX_PATH.exists():
            print("Index not found — building...")
            build_index()

        _index = faiss.read_index(str(INDEX_PATH))
        _catalog = load_catalog()
        return _index, _catalog


def _keyword_score(item: dict, query: str) -> float:
    """Compute a keyword match bonus for an item against a query."""
    query_lower = query.lower()
    query_tokens = set(re.findall(r'\b[a-z0-9#+.]+\b', query_lower))

    score = 0.0
    name_lower = item["name"].lower()
    desc_lower = item.get("description", "").lower()
    combined = name_lower + " " + desc_lower

    # Exact name substring match is strong signal
    for token in query_tokens:
        if len(token) < 2:
            continue
        if token in name_lower:
            score += 0.15
        elif token in desc_lower:
            score += 0.05

    # Test type keyword matching
    type_keywords = {
        "A": {"numerical", "verbal", "reasoning", "cognitive", "aptitude", "ability",
               "logical", "inductive", "deductive", "abstract", "analytical"},
        "P": {"personality", "behavioral", "behaviour", "leadership", "teamwork",
               "motivation", "opq", "cultural", "fit", "communication", "interpersonal"},
        "K": {"knowledge", "programming", "technical", "coding", "software", "developer",
               "engineer", "java", "python", "sql", "c#", ".net", "javascript", "html",
               "css", "react", "angular", "node", "aws", "azure", "devops", "data",
               "network", "linux", "accounting", "finance", "excel", "salesforce"},
        "S": {"simulation", "interactive", "exercise", "situational", "scenario",
               "inbox", "in-tray", "role-play"},
        "C": {"competency", "behavioral competency", "managerial"},
        "B": {"biodata", "situational judgment", "sjt", "judgment"},
    }

    for type_code, keywords in type_keywords.items():
        if query_tokens & keywords and type_code in item.get("test_type", []):
            score += 0.1

    return min(score, 0.4)  # cap keyword bonus


def search(query: str, top_k: int = 15) -> list[dict]:
    """Search the catalog using semantic similarity + keyword boosting."""
    index, catalog = load_index()
    model = get_model()

    query_emb = model.encode([query], normalize_embeddings=True)
    query_emb = np.array(query_emb, dtype="float32")

    # Retrieve more candidates for keyword re-ranking
    n_candidates = min(top_k * 3, len(catalog))
    scores, indices = index.search(query_emb, n_candidates)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = catalog[idx].copy()
        keyword_bonus = _keyword_score(item, query)
        item["score"] = float(score) + keyword_bonus
        results.append(item)

    # Re-sort by combined score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def search_multi(queries: list[str], top_k: int = 15) -> list[dict]:
    """Search with multiple queries and merge results by max score."""
    seen = {}
    for q in queries:
        for item in search(q, top_k):
            key = item["url"]
            if key not in seen or item["score"] > seen[key]["score"]:
                seen[key] = item
    merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


if __name__ == "__main__":
    build_index()
    print("\n--- Test queries ---")
    for query in [
        "Java developer programming test",
        "personality assessment for leadership",
        "numerical reasoning ability test",
        "customer service simulation",
    ]:
        results = search(query, top_k=5)
        print(f"\nQuery: {query}")
        for r in results:
            print(f"  {r['score']:.3f}  {r['name']} ({', '.join(r.get('test_type', []))})")
