# check_booklet_content.py
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
collection_name = "xpcs_documents"

results = client.scroll(
    collection_name=collection_name,
    scroll_filter={
        "must": [
            {
                "key": "source",
                "match": {
                    "text": "x-ray-data-booklet"
                }
            }
        ]
    },
    limit=100,
    with_payload=True,
    with_vectors=False
)

print("="*80)
print("QUALITY CHECK: X-RAY DATA BOOKLET CHUNKS")
print("="*80)

# Categorize chunks by length
good_chunks = []
medium_chunks = []
short_chunks = []

for point in results[0]:
    text = point.payload.get('text', '')
    length = len(text)
    
    if length > 200:
        good_chunks.append(text)
    elif length > 50:
        medium_chunks.append(text)
    else:
        short_chunks.append(text)

print(f"\nChunk quality breakdown (out of 100 samples):")
print(f"  Good (>200 chars): {len(good_chunks)}")
print(f"  Medium (50-200 chars): {len(medium_chunks)}")
print(f"  Short (<50 chars): {len(short_chunks)}")

print(f"\n{'='*80}")
print("SAMPLE OF GOOD CHUNKS:")
print('='*80)

for i, text in enumerate(good_chunks[:5], 1):
    print(f"\n--- Good Chunk {i} ---")
    print(f"Length: {len(text)} chars")
    print(f"Text:\n{text[:400]}")
    print("...")
