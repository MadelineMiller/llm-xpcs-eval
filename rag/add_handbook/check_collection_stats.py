# check_collection_stats.py
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from pathlib import Path
from collections import Counter

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

client = QdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 6333))
)

collection_name = "xpcs_documents"

# Get collection info
collection_info = client.get_collection(collection_name)
print(f"Total points: {collection_info.points_count}")

# Scroll through all points to get statistics
offset = None
doc_types = Counter()
sources = Counter()

print("\nAnalyzing collection...")

while True:
    records, offset = client.scroll(
        collection_name=collection_name,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    
    for record in records:
        doc_type = record.payload.get('document_type', 'xpcs_publication')
        source = record.payload.get('source', 'unknown')
        
        doc_types[doc_type] += 1
        
        # Categorize sources
        if 'xray-data-booklet' in source or 'x-ray-data-booklet' in source:
            sources['X-ray Data Booklet'] += 1
        elif 'textbooks' in source:
            sources['Other Textbooks'] += 1
        else:
            sources['XPCS Publications'] += 1
    
    if offset is None:
        break

print("\n" + "="*60)
print("COLLECTION STATISTICS")
print("="*60)

print("\nDocument Types:")
for doc_type, count in doc_types.most_common():
    print(f"  {doc_type}: {count}")

print("\nSource Categories:")
for source, count in sources.most_common():
    print(f"  {source}: {count}")

print("\n" + "="*60)
