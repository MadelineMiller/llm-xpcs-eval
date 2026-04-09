# rag/citations/verify_qdrant.py
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)


results = client.scroll(
    collection_name="xpcs_documents",
    limit=3,
    with_payload=True,
    with_vectors=False
)


for point in results[0]:
    p = point.payload
    print("---")
    print(f"  source:  {p.get('source')}")
    print(f"  page:    {p.get('page')}")
    print(f"  title:   {p.get('title')}")
    print(f"  journal: {p.get('journal')}")
    print(f"  year:    {p.get('year')}")
    print(f"  authors: {p.get('authors')}")
    print(f"  doi:     {p.get('doi')}")
    print(f"  url:     {p.get('url')}")
    