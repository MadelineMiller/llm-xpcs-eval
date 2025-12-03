from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import os

load_dotenv()

def search_documents(query: str, top_k: int = 3):
    """Search for relevant documents."""
    
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="allenai/scibert_scivocab_uncased",
        model_kwargs={'device': 'cpu'}
    )
    
    print("Connecting to Qdrant...")
    client = QdrantClient(
        host=os.getenv('QDRANT_HOST', 'localhost'),
        port=int(os.getenv('QDRANT_PORT', 6333))
    )
    
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')
    
    print(f"Searching for: '{query}'\n")
    
    # Generate query embedding
    query_vector = embeddings.embed_query(query)
    
    # Search using the correct method
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    )
    
    print("=" * 80)
    
    for idx, result in enumerate(results.points, 1):
        print(f"\nResult {idx} (Similarity Score: {result.score:.4f})")
        print(f"Source: {os.path.basename(result.payload['source'])}")
        print(f"Page: {result.payload['page']}")
        print(f"\nText:\n{result.payload['text'][:400]}...")
        print("-" * 80)
    
    return results

if __name__ == "__main__":
    # Test queries
    queries = [
        "What is X-ray Photon Correlation Spectroscopy?",
        "What are the sample requirements for XPCS experiments?",
    ]
    
    for query in queries:
        search_documents(query, top_k=2)
        print("\n" + "=" * 80 + "\n")
