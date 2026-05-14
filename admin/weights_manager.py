import json
import os
from qdrant_client import QdrantClient

WEIGHTS_FILE = "doc_weights.json"

def load_weights() -> dict:
    if not os.path.exists(WEIGHTS_FILE):
        return {}
    with open(WEIGHTS_FILE, "r") as f:
        return json.load(f)

def save_weights(weights: dict):
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(weights, f, indent=2)

def get_all_docs(client: QdrantClient, collection_name: str) -> list[dict]:
    """Get one entry per unique document from Qdrant."""
    seen = set()
    docs = []
    offset = None

    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        for point in results:
            p = point.payload
            source = os.path.basename(p.get("source", "unknown"))
            title = p.get("title") or source

            if source not in seen:
                seen.add(source)
                docs.append({
                    "source":  source,
                    "title":   title,
                    "authors": p.get("authors") or [],
                    "year":    p.get("year") or "",
                    "journal": p.get("journal") or "",
                    "doi":     p.get("doi") or "",
                    "url":     p.get("url") or "",
                })

        if offset is None:
            break

    return sorted(docs, key=lambda x: x["title"].lower())

def apply_weights(results, weights: dict) -> list:
    """Sort results by weight (primary) then similarity (secondary)."""
    scored = []
    for result in results:
        source     = os.path.basename(result.payload.get("source", ""))
        raw_weight = weights.get(source, 50)

        # Weight 0 = exclude entirely
        if raw_weight == 0:
            print(f"  [WEIGHT] {source}: EXCLUDED (weight=0)")
            continue

        print(f"  [WEIGHT] {source}: "
              f"similarity={result.score:.4f} | weight={raw_weight}/100")

        scored.append((raw_weight, result.score, result))

    # Sort by weight descending, then similarity descending as tiebreaker
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [r for _, _, r in scored]
