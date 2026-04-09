# rag/citations/audit_qdrant.py -- replace with this
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

results = client.scroll(
    collection_name="xpcs_documents",
    limit=5,
    with_payload=True,
    with_vectors=False
)

for point in results[0]:
    print(point.payload)
    print("---")
