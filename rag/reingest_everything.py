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

load_dotenv()

def ingest_all_documents():
    """Load ALL PDFs from textbooks and XPCS publications directories."""
    
    # Define both directories - UPDATED FOR YOUR NEW STRUCTURE
    pdf_sources = [
        {
            'dir': './context/context_docs/xpcs_publications',
            'description': 'XPCS Research Papers',
            'pattern': '*.pdf'
        },
        {
            'dir': './context/context_docs/textbooks',
            'description': 'Reference Textbooks',
            'pattern': '*.pdf'
        }
    ]
    
    all_pdf_files = []
    
    # Collect all PDFs
    print("="*70)
    print("SCANNING FOR PDF FILES")
    print("="*70)
    
    for source in pdf_sources:
        pdf_dir = source['dir']
        if os.path.exists(pdf_dir):
            # Get all PDFs from the directory
            pdfs = [f for f in Path(pdf_dir).glob(source['pattern']) if f.is_file()]
            all_pdf_files.extend(pdfs)
            print(f"✅ {source['description']}: {len(pdfs)} PDFs")
            if len(pdfs) > 0:
                print(f"   Sample: {pdfs[0].name}")
        else:
            print(f"❌ Directory not found: {pdf_dir}")
    
    print(f"\n📚 Total PDFs to process: {len(all_pdf_files)}")
    
    if not all_pdf_files:
        print("\n❌ ERROR: No PDF files found!")
        return
    
    # Show what will be processed
    print("\n📋 Files to be ingested:")
    print("   Textbooks:")
    for pdf in sorted(all_pdf_files):
        if 'textbooks' in str(pdf):
            print(f"     - {pdf.name}")
    
    xpcs_count = sum(1 for p in all_pdf_files if 'xpcs_publications' in str(p))
    print(f"\n   XPCS Papers: {xpcs_count} files")
    
    # Confirm before proceeding
    print("\n" + "="*70)
    print("⚠️  WARNING: This will REPLACE your entire Qdrant collection")
    print(f"    New database will contain: {len(all_pdf_files)} PDFs")
    print(f"    - {xpcs_count} XPCS research papers")
    print(f"    - {len(all_pdf_files) - xpcs_count} textbooks")
    print("="*70)
    response = input("\nContinue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    # Load PDFs
    all_documents = []
    failed_files = []
    
    print("\n" + "="*70)
    print("LOADING PDF DOCUMENTS")
    print("="*70)
    
    for pdf_path in tqdm(all_pdf_files, desc="Loading PDFs"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            failed_files.append((pdf_path.name, str(e)))
            tqdm.write(f"❌ Failed: {pdf_path.name}")
            continue
    
    print(f"\n✅ Loaded {len(all_documents)} pages from {len(all_pdf_files) - len(failed_files)}/{len(all_pdf_files)} PDFs")
    
    if not all_documents:
        print("❌ ERROR: No documents were successfully loaded")
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
    chunks = text_splitter.split_documents(all_documents)
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
    
    # Connect to Qdrant
    print("\n" + "="*70)
    print("CONNECTING TO QDRANT")
    print("="*70)
    
    client = QdrantClient(
        host=os.getenv('QDRANT_HOST', 'localhost'),
        port=int(os.getenv('QDRANT_PORT', 6333))
    )
    
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')
    
    # Delete and recreate collection
    print(f"\n⚠️  Recreating collection '{collection_name}'...")
    try:
        client.delete_collection(collection_name)
        print("✅ Deleted old collection")
    except:
        print("ℹ️  No existing collection to delete")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    print(f"✅ Created fresh collection")
    
    # Generate embeddings and upload
    print("\n" + "="*70)
    print("GENERATING EMBEDDINGS & UPLOADING")
    print("="*70)
    print("⏱️  This will take 15-20 minutes...")
    print("    Perfect time for a coffee break! ☕\n")
    
    points = []
    batch_size = 100
    
    for idx, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        embedding = embeddings.embed_query(chunk.page_content)
        
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
        
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            points = []
    
    # Upload remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    # Verify
    collection_info = client.get_collection(collection_name)
    
    # Summary
    print("\n" + "="*70)
    print("✅ INGESTION COMPLETE")
    print("="*70)
    print(f"Total PDFs processed: {len(all_pdf_files)}")
    print(f"Successfully loaded: {len(all_pdf_files) - len(failed_files)}")
    print(f"Failed to load: {len(failed_files)}")
    print(f"Total pages: {len(all_documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Vectors in database: {collection_info.points_count}")
    print("="*70)
    
    if failed_files:
        print("\n⚠️  Failed files:")
        for fname, _ in failed_files:
            print(f"  - {fname}")
    
    print("\n✅ Your database now contains:")
    print(f"  - {xpcs_count} XPCS research papers")
    print(f"  - {len(all_pdf_files) - xpcs_count} reference textbooks")
    print("\n🎯 Next steps:")
    print("  1. Run 'python check_handbook.py' to verify")
    print("  2. Run 'chainlit run app.py' to test your chatbot")
    print("  3. Ask 'What is XPCS?' and see the difference!")

if __name__ == "__main__":
    ingest_all_documents()
