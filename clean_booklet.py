from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
collection_name = "xpcs_documents"

print("="*70)
print("CLEANING X-RAY DATA BOOKLET CHUNKS")
print("="*70)

print("\nFetching all booklet chunks...")
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
    limit=2000,
    with_payload=True,
    with_vectors=False
)

print(f"Found {len(results[0])} total booklet chunks")

bad_ids = []
good_chunks = []
medium_chunks = []

for point in results[0]:
    text = point.payload.get('text', '')
    length = len(text)
    
    if length < 100:
        bad_ids.append(point.id)
    elif length > 200:
        good_chunks.append(text)
    else:
        medium_chunks.append(text)

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print(f"Bad chunks (<100 chars): {len(bad_ids)}")
print(f"Medium chunks (100-200 chars): {len(medium_chunks)}")
print(f"Good chunks (>200 chars): {len(good_chunks)}")
print(f"\nWill delete: {len(bad_ids)} chunks")
print(f"Will keep: {len(good_chunks) + len(medium_chunks)} chunks")

print("\n" + "="*70)
print("SAMPLE OF CHUNKS THAT WILL BE KEPT:")
print("="*70)
for i, text in enumerate(good_chunks[:3], 1):
    print(f"\n--- Sample {i} ---")
    print(f"Length: {len(text)} chars")
    print(f"Text: {text[:300]}...")

print("\n" + "="*70)
response = input("\nDelete bad chunks? (yes/no): ")

if response.lower() == 'yes':
    print("\nDeleting bad chunks...")
    client.delete(
        collection_name=collection_name,
        points_selector=bad_ids
    )
    print(f"Deleted {len(bad_ids)} bad chunks")
    
    new_info = client.get_collection(collection_name)
    print(f"\nCurrent total: {new_info.points_count}")
else:
    print("Cancelled.")
