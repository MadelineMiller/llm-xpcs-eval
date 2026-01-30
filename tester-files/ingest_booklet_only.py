# ingest_booklet_only.py
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from dotenv import load_dotenv
import os
from tqdm import tqdm
import uuid

load_dotenv()

def ingest_single_pdf_with_ocr():
    """Ingest ONLY the X-ray Data Booklet using OCR and ADD to existing collection."""
    
    pdf_path = "./context/context_docs/textbooks/x-ray-data-booklet.pdf"
    
    print("="*70)
    print("INGESTING X-RAY DATA BOOKLET WITH OCR")
    print("="*70)
    print(f"📄 File: {pdf_path}")
    print("⚠️  This will take 30-60 minutes")
    print("✅ Will ADD to existing collection (not replace)")
    print("="*70)
    
    # Connect to Qdrant FIRST to verify collection exists
    print("\n" + "="*70)
    print("CONNECTING TO QDRANT")
    print("="*70)
    
    client = QdrantClient(
        host=os.getenv('QDRANT_HOST', 'localhost'),
        port=int(os.getenv('QDRANT_PORT', 6333))
    )
    
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')
    
    # Check collection exists
    try:
        collection_info = client.get_collection(collection_name)
        print(f"✅ Found existing collection: {collection_name}")
        print(f"   Current vectors: {collection_info.points_count}")
    except Exception as e:
        print(f"❌ ERROR: Collection '{collection_name}' not found!")
        print(f"   {e}")
        return
    
    # SAFETY CHECK
    print("\n" + "="*70)
    print("⚠️  SAFETY CONFIRMATION")
    print("="*70)
    print(f"Current collection has: {collection_info.points_count} vectors")
    print(f"\nThis script will:")
    print(f"  1. Remove old booklet entries (if any)")
    print(f"  2. Add new booklet chunks with OCR")
    print(f"  3. Keep ALL existing vectors from other PDFs")
    print(f"\n✅ YOUR XPCS PAPERS WILL NOT BE TOUCHED")
    print("="*70)
    
    response = input("\nType 'ADD BOOKLET' to confirm: ")
    if response != "ADD BOOKLET":
        print("Cancelled for safety.")
        return
    
    # First, DELETE old booklet entries if they exist
    print("\n" + "="*70)
    print("REMOVING OLD BOOKLET ENTRIES (if any)")
    print("="*70)
    
    try:
        # Get all points from the booklet
        old_results = client.scroll(
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
            limit=10000,
            with_payload=False,
            with_vectors=False
        )
        
        old_ids = [point.id for point in old_results[0]]
        
        if old_ids:
            client.delete(
                collection_name=collection_name,
                points_selector=old_ids
            )
            print(f"✅ Deleted {len(old_ids)} old booklet entries")
        else:
            print("ℹ️  No old booklet entries found")
            
    except Exception as e:
        print(f"⚠️  Could not delete old entries: {e}")
    
    # Load PDF with OCR
    print("\n" + "="*70)
    print("LOADING PDF WITH OCR")
    print("="*70)
    print("🔍 Starting OCR... (grab a coffee ☕)")
    print("   This will take 30-60 minutes for 176 pages")
    
    try:
        loader = UnstructuredPDFLoader(
            pdf_path,
            mode="elements",
            strategy="hi_res"  # Use OCR
        )
        docs = loader.load()
        print(f"\n✅ OCR complete! Extracted {len(docs)} elements")
        
        # Show sample
        if docs:
            sample_text = docs[0].page_content[:200]
            print(f"\nSample text:\n{sample_text}...")
            
    except Exception as e:
        print(f"❌ OCR failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure tesseract is installed: conda list tesseract")
        print("  2. Make sure Python packages are installed:")
        print("     pip install unstructured pdf2image pytesseract pillow")
        return
    
    if not docs:
        print("❌ No content extracted!")
        return
    
    # Split into chunks
    print("\n" + "="*70)
    print("SPLITTING INTO CHUNKS")
    print("="*70)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Load embeddings model
    print("\n" + "="*70)
    print("LOADING EMBEDDINGS MODEL")
    print("="*70)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="allenai/scibert_scivocab_uncased",
        model_kwargs={'device': 'cpu'}
    )
    print("✅ SciBERT model loaded")
    
    # Generate embeddings and upload
    print("\n" + "="*70)
    print("GENERATING EMBEDDINGS & UPLOADING")
    print("="*70)
    print("⏱️  This will take 10-15 minutes...")
    
    points = []
    batch_size = 100
    
    for idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        embedding = embeddings.embed_query(chunk.page_content)
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", pdf_path),
                "page": chunk.metadata.get("page", 0),
            }
        )
        points.append(point)
        
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            points = []
    
    # Upload remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    # Verify
    new_collection_info = client.get_collection(collection_name)
    
    # Summary
    print("\n" + "="*70)
    print("✅ INGESTION COMPLETE")
    print("="*70)
    print(f"Previous vectors: {collection_info.points_count}")
    print(f"New booklet chunks: {len(chunks)}")
    print(f"Total vectors now: {new_collection_info.points_count}")
    print(f"Net change: +{new_collection_info.points_count - collection_info.points_count}")
    print("="*70)
    
    print("\n🎯 Next step: Test with your chatbot!")
    print("   Ask: 'What is the atomic scattering factor for carbon?'")
    print("   Or: 'Tell me about the X-ray Data Booklet'")

if __name__ == "__main__":
    ingest_single_pdf_with_ocr()
