"""Show the actual text content of Perakis paper chunks to diagnose keyword matching."""
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
    for point in results:
        src = point.payload.get('source', '')
        if 'perakis' in src.lower() or 'diffusive' in src.lower() or 'amorphous ice 1' in src.lower():
            found.append(point)
    if offset is None:
        break

print("Found", len(found), "Perakis chunks\n")

# Show chunks that mention variance or autocorrelation
keywords = ['variance', 'autocorrelation', 'auto-correlation', 'normalized', 'temporal']
for point in found:
    text = point.payload.get('text', '')
    text_lower = text.lower()
    if any(kw in text_lower for kw in keywords):
        print("--- page", point.payload.get('page'), "---")
        print(text[:800])
        print()
