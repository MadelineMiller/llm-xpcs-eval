# rag/citations/diagnose_problems.py
import fitz
import re

problem_files = {
    "057_Byrne_2008.pdf":    "context/context_docs/xpcs_publications/057_Byrne_2008.pdf",
    "062_Ruta_2012.pdf":     "context/context_docs/xpcs_publications/062_Ruta_2012.pdf",
    "063_Evenson_2015.pdf":  "context/context_docs/xpcs_publications/063_Evenson_2015.pdf",
    "068_Trappe_2007.pdf":   "context/context_docs/xpcs_publications/068_Trappe_2007.pdf",
    "103_Grybos_2016.pdf":   "context/context_docs/xpcs_publications/103_Grybos_2016.pdf",
    "104_Zhang_2017.pdf":    "context/context_docs/xpcs_publications/104_Zhang_2017.pdf",
}

doi_pattern = r'10\.\d{4,}/[^\s\]>"\'،,]+'

for name, path in problem_files.items():
    print(f"\n{'='*50}")
    print(f"FILE: {name}")
    doc = fitz.open(path)

    print("  PDF metadata:")
    for k, v in doc.metadata.items():
        if v:
            print(f"    {k}: {v}")

    print("  First page text (first 800 chars):")
    text = doc[0].get_text()
    print(text[:800])

    print("  All DOI matches found in first 2 pages:")
    for i in range(min(2, len(doc))):
        matches = re.findall(doi_pattern, doc[i].get_text())
        for m in matches:
            print(f"    page {i}: {m}")
