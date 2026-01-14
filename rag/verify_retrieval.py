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

print("🔍 Verifying XPCS Database Contents...")
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
textbook_passages = {
    'elements-of-modern-x-ray.pdf': [],
    'hard-xray-photon.pdf': [],
    'x-ray-data-booklet.pdf': []
}

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
        basename = os.path.basename(source)
        all_sources.append(basename)
        
        # Check for textbooks
        for textbook in textbook_passages.keys():
            if textbook in basename:
                textbook_passages[textbook].append({
                    'page': point.payload.get('page', 'N/A'),
                    'text': point.payload.get('text', '')[:100]
                })
    
    if offset is None:
        break

# Count documents
doc_counts = Counter(all_sources)

# Results
print("\n" + "="*70)
print("📊 DATABASE VERIFICATION RESULTS")
print("="*70)

# Check textbooks
print("\n📚 TEXTBOOKS:")
for textbook, passages in textbook_passages.items():
    if passages:
        print(f"   ✅ {textbook}: {len(passages)} passages")
    else:
        print(f"   ❌ {textbook}: NOT FOUND")

# Count XPCS papers
xpcs_papers = [doc for doc in doc_counts.keys() if doc.startswith(tuple(f'{i:03d}_' for i in range(1, 200)))]
print(f"\n📄 XPCS RESEARCH PAPERS:")
print(f"   ✅ Found {len(xpcs_papers)} papers with {sum(doc_counts[doc] for doc in xpcs_papers)} total passages")

# Summary statistics
print("\n" + "="*70)
print("📈 SUMMARY STATISTICS")
print("="*70)
print(f"Total unique documents: {len(doc_counts)}")
print(f"Total passages: {sum(doc_counts.values())}")
print(f"Textbooks: {sum(1 for t, p in textbook_passages.items() if p)}/3")
print(f"XPCS papers: {len(xpcs_papers)}")

# Show sample of XPCS papers
print("\n📋 Sample XPCS Papers (first 10):")
for doc in sorted(xpcs_papers)[:10]:
    print(f"   - {doc}: {doc_counts[doc]} passages")

# Show textbook samples
print("\n📖 Textbook Sample Passages:")
for textbook, passages in textbook_passages.items():
    if passages:
        print(f"\n   {textbook} (Page {passages[0]['page']}):")
        print(f"   {passages[0]['text']}...")

# Final verdict
print("\n" + "="*70)
if len(xpcs_papers) >= 100 and all(passages for passages in textbook_passages.values()):
    print("✅ DATABASE VERIFICATION PASSED!")
    print("   Your RAG system is ready to use!")
else:
    print("⚠️  DATABASE INCOMPLETE")
    if len(xpcs_papers) < 100:
        print(f"   - Expected ~114 XPCS papers, found {len(xpcs_papers)}")
    for textbook, passages in textbook_passages.items():
        if not passages:
            print(f"   - Missing: {textbook}")
print("="*70)
