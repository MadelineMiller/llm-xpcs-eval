"""
One-time migration: adds a full-text payload index to the 'text' field
in the Qdrant collection. Run this once, then restart the app.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import TextIndexParams, TokenizerType
from dotenv import load_dotenv
import os

load_dotenv()
client = QdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 6333))
)
collection = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')

print("Adding full-text index to '{}' field in collection '{}'...".format("text", collection))

client.create_payload_index(
    collection_name=collection,
    field_name="text",
    field_schema=TextIndexParams(
        type="text",
        tokenizer=TokenizerType.WORD,
        min_token_len=2,
        max_token_len=50,
        lowercase=True,
    )
)

print("Done. Hybrid retrieval (vector + keyword) is now available.")
