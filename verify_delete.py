# verify_delete.py
from qdrant_client import QdrantClient
import os

client = QdrantClient(host="localhost", port=6333)
collection = "xpcs_documents"
filename = "059_Berthier_2005_old.pdf"

count = 0
offset = None
while True:
    results, offset = client.scroll(
        collection_name=collection,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    for point in results:
        source = os.path.basename(point.payload.get("source", ""))
        if source == filename:
            count += 1
            print(f"  FOUND: point_id={point.id}")

    if offset is None:
        break

if count == 0:
    print(f"✅ No chunks found for {filename} — fully deleted!")
else:
    print(f"⚠️ {count} chunks still exist for {filename}")
