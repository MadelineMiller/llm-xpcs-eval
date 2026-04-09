# rag/citations/update_qdrant_payloads.py
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

COLLECTION  = "xpcs_documents"
MAP_FILE    = "rag/citations/metadata_map.json"

client = QdrantClient(host="localhost", port=6333)

with open(MAP_FILE) as f:
    metadata_map = json.load(f)

skipped  = 0
updated  = 0
no_title = 0

for filename, meta in sorted(metadata_map.items()):

    # skip the old duplicate
    if meta.get("SKIP"):
        print(f"  ⏭️  SKIP: {filename}")
        skipped += 1
        continue

    if not meta.get("title"):
        print(f"  ⚠️  no title, skipping: {filename}")
        no_title += 1
        continue

    # build the payload fields to add
    payload = {
        "title":   meta.get("title", ""),
        "journal": meta.get("journal", ""),
        "year":    meta.get("year", ""),
        "authors": meta.get("authors", []),
        "doi":     meta.get("doi", ""),
        "url":     meta.get("url", ""),
    }

    # match all Qdrant points whose source ends with this filename
    # source is stored as full path e.g. context/context_docs/xpcs_publications/001_Livet_2007.pdf
    try:
        client.set_payload(
            collection_name=COLLECTION,
            payload=payload,
            points=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=f"context/context_docs/xpcs_publications/{filename}")
                    )
                ]
            )
        )
        print(f"  ✅ {filename}  →  {meta['title'][:60]}")
        updated += 1

    except Exception as e:
        print(f"  ❌ ERROR on {filename}: {e}")

print(f"\n{'='*50}")
print(f"Updated:  {updated}")
print(f"Skipped:  {skipped}")
print(f"No title: {no_title}")
