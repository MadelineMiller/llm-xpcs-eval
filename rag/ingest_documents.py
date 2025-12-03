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
from config import RETRIEVAL_CONFIG

load_dotenv()

def ingest_documents():
    """Load PDFs, chunk them, embed them, and store in Qdrant."""
    
    pdf_dir = "context_docs/xpcs_publications"
    
    # Verify directory exists
    if not os.path.exists(pdf_dir):
        print(f"ERROR: Directory not found: {pdf_dir}")
        return
    
    # Get all PDF files
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files\n")
    
    # Load PDFs one by one with error handling
    all_documents = []
    failed_files = []
    
    print("Loading PDF documents...")
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            failed_files.append((pdf_path.name, str(e)))
            print(f"\nWARNING: Failed to load {pdf_path.name}: {str(e)[:100]}")
            continue
    
    print(f"\nSuccessfully loaded {len(all_documents)} pages from {len(pdf_files) - len(failed_files)} PDFs")
    if failed_files:
        print(f"Failed to load {len(failed_files)} PDFs:")
        for fname, error in failed_files:
            print(f"  - {fname}")
    
    if not all_documents:
        print("ERROR: No documents were successfully loaded")
        return
    
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RETRIEVAL_CONFIG['chunk_size'],
        chunk_overlap=RETRIEVAL_CONFIG['chunk_overlap'],
        length_function=len,
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} chunks")
    
    print("\nLoading SciBERT embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="allenai/scibert_scivocab_uncased",
        model_kwargs={'device': 'cpu'}
    )
    print("Model loaded")
    
    print("\nConnecting to Qdrant...")
    client = QdrantClient(
        host=os.getenv('QDRANT_HOST', 'localhost'),
        port=int(os.getenv('QDRANT_PORT', 6333))
    )
    
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')
    
    # Delete and recreate collection for fresh start
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    print(f"Created collection: {collection_name}")
    
    print("\nGenerating embeddings and uploading to Qdrant...")
    print("This may take several minutes depending on the number of documents...")
    
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
    
    print(f"\nSuccessfully ingested {len(chunks)} chunks into Qdrant")
    
    # Verify
    collection_info = client.get_collection(collection_name)
    print(f"Collection stats: {collection_info.points_count} points")
    
    # Summary
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    print(f"Total PDFs found: {len(pdf_files)}")
    print(f"Successfully loaded: {len(pdf_files) - len(failed_files)}")
    print(f"Failed to load: {len(failed_files)}")
    print(f"Total pages: {len(all_documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Vectors in database: {collection_info.points_count}")
    print("="*60)

if __name__ == "__main__":
    ingest_documents()
