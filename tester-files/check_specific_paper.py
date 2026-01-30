# Check if Chu et al. 2023 is in the database
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Search for "Lloyd's mirror" or "Chu" or "holography"
results = client.scroll(
    collection_name="xpcs_documents",
    scroll_filter={
        "must": [
            {
                "key": "text",
                "match": {
                    "text": "Lloyd"
                }
            }
        ]
    },
    limit=10,
    with_payload=True
)

print(f"Found {len(results[0])} results for 'Lloyd'")
for point in results[0]:
    print(f"Source: {point.payload['source']}")
    print(f"Text: {point.payload['text'][:200]}...")
    print()