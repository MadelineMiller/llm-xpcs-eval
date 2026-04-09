# rag/citations/check_textbook.py
import fitz

doc = fitz.open("context/context_docs/textbooks/hard-xray-photon.pdf")
print("METADATA:", doc.metadata)
print("\nFIRST PAGE:")
print(doc[0].get_text()[:600])