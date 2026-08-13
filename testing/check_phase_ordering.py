from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()
client = QdrantClient(host=os.getenv('QDRANT_HOST', 'localhost'), port=int(os.getenv('QDRANT_PORT', 6333)))
collection = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')

offset = None
found = []

while True:
    results, offset = client.scroll(
        collection_name=collection,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    for p in results:
        title  = p.payload.get('title', '')
        source = p.payload.get('source', '')
        if 'phase' in title.lower() or 'phase' in source.lower() or 'xifs' in title.lower():
            found.append((source, title))
    if offset is None:
        break

seen = set()
for src, ttl in found:
    if src not in seen:
        print(src, '|', ttl[:80])
        seen.add(src)

if not seen:
    print("No matching documents found.")
