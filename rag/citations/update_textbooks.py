# rag/citations/update_textbooks.py
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

COLLECTION = "xpcs_documents"


client = QdrantClient(host="localhost", port=6333)


textbooks = {
    "elements-of-modern-x-ray.pdf": {
        "title":   "Elements of Modern X-ray Physics",
        "journal": "Wiley",
        "year":    2011,
        "authors": ["Jens Als-Nielsen", "Des McMorrow"],
        "doi":     "10.1002/9781119998365",
        "url":     "https://doi.org/10.1002/9781119998365"
    },
    "hard-xray-photon.pdf": {
        "title":   "Hard X-Ray Photon Correlation Spectroscopy Methods for Materials Studies",
        "journal": "Annual Review of Materials Research",
        "year":    2018,
        "authors": ["Alec R. Sandy", "Qingteng Zhang", "Laurence B. Lurio"],
        "doi":     "10.1146/annurev-matsci-070317-124358",
        "url":     "https://doi.org/10.1146/annurev-matsci-070317-124358"
    },
    "x-ray-data-booklet.pdf": {
        "title":   "X-Ray Data Booklet",
        "journal": "Lawrence Berkeley National Laboratory",
        "year":    2009,
        "authors": ["Albert C. Thompson"],
        "doi":     "",
        "url":     "https://xdb.lbl.gov/"
    },
    "xray-data-booklet-local.pdf": {
        "title":   "X-Ray Data Booklet",
        "journal": "Lawrence Berkeley National Laboratory",
        "year":    2009,
        "authors": ["Albert C. Thompson"],
        "doi":     "",
        "url":     "https://xdb.lbl.gov/"
    },
}


for filename, meta in textbooks.items():
    source_path = f"context/context_docs/textbooks/{filename}"
    try:
        client.set_payload(
            collection_name=COLLECTION,
            payload=meta,
            points=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_path)
                    )
                ]
            )
        )
        print(f"  ✅ {filename}  →  {meta['title']}")
    except Exception as e:
        print(f"  ❌ ERROR on {filename}: {e}")
        