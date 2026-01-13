from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

# Connect to Qdrant
client = QdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 6333))
)

collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')

print("🔍 Checking for x-ray_data_handbook.pdf in Qdrant...")
print("="*70)

# Get collection info
try:
    collection_info = client.get_collection(collection_name)
    print(f"✅ Collection '{collection_name}' found")
    print(f"   Total points: {collection_info.points_count}")
except Exception as e:
    print(f"❌ Error accessing collection: {e}")
    exit(1)

# Scroll through all points
print("\n📚 Scanning all documents...")
offset = None
all_sources = []
handbook_passages = []

while True:
    result = client.scroll(
        collection_name=collection_name,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    
    points, offset = result
    
    if not points:
        break
    
    for point in points:
        source = point.payload.get('source', '')
        all_sources.append(os.path.basename(source))
        
        # Check for handbook
        if 'x-ray_data_handbook' in source.lower() or 'xray_data_handbook' in source.lower():
            handbook_passages.append({
                'page': point.payload.get('page', 'N/A'),
                'text': point.payload.get('text', '')[:150]
            })
    
    if offset is None:
        break

# Results
print("\n" + "="*70)
print("📊 RESULTS")
print("="*70)

if handbook_passages:
    print(f"✅ x-ray_data_handbook.pdf IS in the database!")
    print(f"   Found {len(handbook_passages)} passages")
    print(f"\n   Sample passages:")
    for i, passage in enumerate(handbook_passages[:3], 1):
        print(f"\n   [{i}] Page {passage['page']}")
        print(f"       {passage['text']}...")
else:
    print("❌ x-ray_data_handbook.pdf NOT FOUND in database")

# Show what IS in the database
print("\n📄 Documents currently in database:")
doc_counts = Counter(all_sources)
for doc, count in sorted(doc_counts.items()):
    print(f"   - {doc}: {count} passages")
