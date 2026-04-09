# test_xray_query.py
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env from project root
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

client = QdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 6333))
)

embeddings = HuggingFaceEmbeddings(
    model_name="allenai/scibert_scivocab_uncased",
    model_kwargs={'device': 'cpu'}
)

# Test query about X-ray properties
query = "What is the atomic scattering factor?"
query_vector = embeddings.embed_query(query)

# Use query_points instead of search
results = client.query_points(
    collection_name="xpcs_documents",
    query=query_vector,
    limit=3
)

print(f"\nQuery: {query}\n")
print("="*80)

for i, result in enumerate(results.points, 1):
    print(f"\n{i}. Score: {result.score:.4f}")
    print(f"   Source: {result.payload['source']}")
    print(f"   Type: {result.payload.get('document_type', 'xpcs_publication')}")
    print(f"   Page: {result.payload.get('page', 'N/A')}")
    print(f"   Text preview: {result.payload['text'][:300]}...")
    print("-"*80)
