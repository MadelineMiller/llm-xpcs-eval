# rag/citations/audit_pdfs.py
import os

PDF_DIR = "context/context_docs/xpcs_publications"

pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
txts = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".txt")])

print(f"PDFs:  {len(pdfs)}")
print(f"TXTs:  {len(txts)}")
print(f"\nAll PDFs:")
for f in pdfs:
    print(f"  {f}")
