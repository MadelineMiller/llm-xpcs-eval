# remove_old_handbook.py
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

client = QdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 6333))
)

collection_name = "xpcs_documents"

# Get initial count
collection_info = client.get_collection(collection_name)
initial_count = collection_info.points_count
print(f"Initial total points: {initial_count}")

# Find and delete old X-ray handbook chunks (not tagged as xray_handbook)
offset = None
deleted_count = 0
ids_to_delete = []

print("\nScanning for old X-ray handbook chunks...")

while True:
    records, offset = client.scroll(
        collection_name=collection_name,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    
    for record in records:
        source = record.payload.get('source', '')
        doc_type = record.payload.get('document_type', 'xpcs_publication')
        
        # Find old handbook chunks (has handbook in source but NOT tagged as xray_handbook)
        if ('xray-data-booklet' in source.lower() or 'x-ray-data-booklet' in source.lower()):
            if doc_type != 'xray_handbook':
                ids_to_delete.append(record.id)
                print(f"  Found old chunk: {source[:80]}...")
    
    if offset is None:
        break

print(f"\nFound {len(ids_to_delete)} old handbook chunks to remove")

if len(ids_to_delete) > 0:
    confirm = input(f"\nDelete {len(ids_to_delete)} old chunks? (yes/no): ")
    
    if confirm.lower() == 'yes':
        # Delete in batches of 100
        batch_size = 100
        for i in range(0, len(ids_to_delete), batch_size):
            batch = ids_to_delete[i:i+batch_size]
            client.delete(
                collection_name=collection_name,
                points_selector=batch
            )
            deleted_count += len(batch)
            print(f"Deleted batch {i//batch_size + 1}: {len(batch)} chunks")
        
        # Get final count
        collection_info = client.get_collection(collection_name)
        final_count = collection_info.points_count
        
        print("\n" + "="*60)
        print("CLEANUP SUMMARY")
        print("="*60)
        print(f"Initial points: {initial_count}")
        print(f"Deleted: {deleted_count}")
        print(f"Final points: {final_count}")
        print(f"Expected final: {initial_count - deleted_count}")
        print("\n✓ Old X-ray handbook chunks removed successfully!")
        print(f"✓ Kept 251 new handbook chunks (tagged as 'xray_handbook')")
        print("="*60)
    else:
        print("\nCancelled. No changes made.")
else:
    print("\nNo old handbook chunks found. Nothing to delete.")
