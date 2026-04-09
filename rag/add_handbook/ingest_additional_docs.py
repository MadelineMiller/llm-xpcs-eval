from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import os
from tqdm import tqdm
import uuid
from pathlib import Path

# Load .env from project root (two levels up)
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def ingest_additional_documents(pdf_path):
    """Add new PDF to existing Qdrant collection without deleting existing data."""
    
    # Convert to absolute path if relative
    pdf_path = Path(pdf_path).resolve()
    
    # Verify file exists
    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        return
    
    print(f"Loading PDF: {pdf_path}\n")
    
    # Load PDF
    try:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} pages")
    except Exception as e:
        print(f"✗ Failed to load PDF: {e}")
        return
    
    # Split into chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Load embeddings model
    print("\nLoading SciBERT embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="allenai/scibert_scivocab_uncased",
        model_kwargs={'device': 'cpu'}
    )
    print("✓ Model loaded")
    
    # Connect to Qdrant
    print("\nConnecting to Qdrant...")
    client = QdrantClient(
        host=os.getenv('QDRANT_HOST', 'localhost'),
        port=int(os.getenv('QDRANT_PORT', 6333))
    )
    
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')
    
    # Check if collection exists, create if not
    try:
        collection_info = client.get_collection(collection_name)
        points_before = collection_info.points_count
        print(f"✓ Connected to existing collection: {collection_name}")
        print(f"  Current points: {points_before}")
    except:
        print(f"Collection doesn't exist. Creating: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        points_before = 0
    
    # Generate embeddings and upload
    print("\nGenerating embeddings and uploading to Qdrant...")
    
    points = []
    batch_size = 100
    
    for idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        # Generate embedding
        embedding = embeddings.embed_query(chunk.page_content)
        
        # Create point
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                "page": chunk.metadata.get("page", 0),
                "document_type": "xray_handbook"  # Tag for identification
            }
        )
        points.append(point)
        
        # Upload in batches
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            points = []
    
    # Upload remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    # Verify
    collection_info = client.get_collection(collection_name)
    points_after = collection_info.points_count
    points_added = points_after - points_before
    
    print(f"\n✓ Successfully added {points_added} new chunks to Qdrant")
    
    # Summary
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    print(f"PDF file: {pdf_path.name}")
    print(f"Pages loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Points before: {points_before}")
    print(f"Points after: {points_after}")
    print(f"Points added: {points_added}")
    print("="*60)

if __name__ == "__main__":
    # Relative path from add_handbook directory
    pdf_path = "../../context/context_docs/textbooks/xray-data-booklet-local.pdf"
    ingest_additional_documents(pdf_path)
