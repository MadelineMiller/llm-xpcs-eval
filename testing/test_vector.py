from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()
client = QdrantClient(host=os.getenv('QDRANT_HOST', 'localhost'), port=int(os.getenv('QDRANT_PORT', 6333)))
collection = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')

offset = None
found = []
total = 0

while True:
    results, offset = client.scroll(
        collection_name=collection,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    for point in results:
        total += 1
        title = point.payload.get('title', '')
        source = point.payload.get('source', '')
        if 'amorphous' in title.lower() or 'amorphous' in source.lower() or 'ice' in title.lower():
            found.append((source, title, point.payload.get('page')))
    if offset is None:
        break

print('Total chunks in DB:', total)
print('Chunks matching amorphous ice:', len(found))
for src, ttl, pg in found[:5]:
    print(' ', src, '|', ttl[:80], '| page', pg)
