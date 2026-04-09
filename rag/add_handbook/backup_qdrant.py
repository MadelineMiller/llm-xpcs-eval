from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from datetime import datetime
import json

load_dotenv()

def backup_qdrant():
    """Create a snapshot backup of the Qdrant collection."""
    
    client = QdrantClient(
        host=os.getenv('QDRANT_HOST', 'localhost'),
        port=int(os.getenv('QDRANT_PORT', 6333))
    )
    
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')
    
    # Create timestamp for backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_name = f"{collection_name}_backup_{timestamp}"
    
    print(f"Creating snapshot backup: {snapshot_name}")
    
    try:
        # Create snapshot
        snapshot_info = client.create_snapshot(collection_name=collection_name)
        print(f"✓ Snapshot created: {snapshot_info.name}")
        
        # Get collection info for metadata
        collection_info = client.get_collection(collection_name)
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "collection_name": collection_name,
            "points_count": collection_info.points_count,
            "snapshot_name": snapshot_info.name
        }
        
        metadata_file = f"backups/metadata_{timestamp}.json"
        os.makedirs("backups", exist_ok=True)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved to: {metadata_file}")
        print(f"\nBackup Summary:")
        print(f"  Collection: {collection_name}")
        print(f"  Points: {collection_info.points_count}")
        print(f"  Snapshot: {snapshot_info.name}")
        
        return snapshot_info.name
        
    except Exception as e:
        print(f"✗ Error creating backup: {e}")
        return None

if __name__ == "__main__":
    backup_qdrant()
