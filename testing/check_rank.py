from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
client = QdrantClient(host=os.getenv('QDRANT_HOST', 'localhost'), port=int(os.getenv('QDRANT_PORT', 6333)))
collection = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')

embeddings = HuggingFaceEmbeddings(
    model_name="allenai/scibert_scivocab_uncased",
    model_kwargs={'device': 'cpu'}
)

query = "What is the normalized variance chi T of the temporal autocorrelation function"
vector = embeddings.embed_query(query)

results = client.query_points(collection_name=collection, query=vector, limit=200, with_payload=True)

print("Searching top 200 results for Perakis / amorphous ice chunks...\n")
for i, r in enumerate(results.points, 1):
    src = os.path.basename(r.payload.get('source', ''))
    if 'amorphous' in src.lower() or 'perakis' in src.lower() or 'diffusive' in src.lower():
        print('Rank', i, '| score', round(r.score, 4), '|', src)
