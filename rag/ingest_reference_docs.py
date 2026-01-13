from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from dotenv import load_dotenv
import os
from tqdm import tqdm
import uuid
from pathlib import Path

load_dotenv()

def ingest_reference_documents():
    """Load reference PDFs, chunk them, embed them, and ADD to existing Qdrant collection."""
    
    pdf_dir = "../context/context_docs"  # Changed to context_docs directory
    
    # Verify directory exists
    if not os.path.exists(pdf_dir):
        print(f"ERROR: Directory not found: {pdf_dir}")
        return
    
    # Get all PDF files
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    print()
    
    # Load PDFs one by one with error handling
    all_documents = []
    failed_files = []
    
    print("Loading PDF documents...")
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            all_documents.extend(docs)
            print(f"\n  ✅ Loaded {pdf_path.name}: {len(docs)} pages")
        except Exception as e:
            failed_files.append((pdf_path.name, str(e)))
            print(f"\n  ❌ Failed to load {pdf_path.name}: {str(e)[:100]}")
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
        chunk_size=1000,
        chunk_overlap=200,
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
    
    # Get current collection stats (DO NOT DELETE)
    try:
        collection_info = client.get_collection(collection_name)
        print(f"✅ Connected to existing collection: {collection_name}")
        print(f"   Current points in database: {collection_info.points_count}")
    except Exception as e:
        print(f"❌ Error: Collection '{collection_name}' not found: {e}")
        print("   Please run the original ingestion script first to create the collection.")
        return
    
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
    
    print(f"\n✅ Successfully added {len(chunks)} chunks to Qdrant")
    
    # Verify
    collection_info_after = client.get_collection(collection_name)
    new_points = collection_info_after.points_count - collection_info.points_count
    
    # Summary
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    print(f"Total PDFs found: {len(pdf_files)}")
    print(f"Successfully loaded: {len(pdf_files) - len(failed_files)}")
    print(f"Failed to load: {len(failed_files)}")
    print(f"Total pages: {len(all_documents)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"\nDatabase stats:")
    print(f"  Points before: {collection_info.points_count}")
    print(f"  Points after:  {collection_info_after.points_count}")
    print(f"  New points added: {new_points}")
    print("="*60)

if __name__ == "__main__":
    ingest_reference_documents()
