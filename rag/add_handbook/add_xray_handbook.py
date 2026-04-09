from backup_qdrant import backup_qdrant
from ingest_additional_docs import ingest_additional_documents
import sys

def main():
    """Backup existing database and add X-ray handbook."""
    
    print("="*60)
    print("STEP 1: Creating Backup")
    print("="*60)
    
    snapshot_name = backup_qdrant()
    
    if not snapshot_name:
        print("\n✗ Backup failed. Aborting ingestion.")
        sys.exit(1)
    
    print("\n✓ Backup completed successfully!")
    print(f"Snapshot: {snapshot_name}")
    
    print("\n" + "="*60)
    print("STEP 2: Adding X-ray Data Handbook")
    print("="*60)
    
    # Correct path relative to add_handbook directory
    pdf_path = "../../context/context_docs/textbooks/xray-data-booklet-local.pdf"
    
    try:
        ingest_additional_documents(pdf_path)
        print("\n✓ X-ray handbook successfully added!")
    except Exception as e:
        print(f"\n✗ Error adding handbook: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print(f"Backup snapshot: {snapshot_name}")
    print("X-ray handbook successfully added to database")
    print("="*60)

if __name__ == "__main__":
    main()
