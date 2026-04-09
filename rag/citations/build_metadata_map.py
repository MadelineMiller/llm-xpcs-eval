# rag/citations/build_metadata_map.py
import os
import re
import json
import time
import fitz  # PyMuPDF
import requests

PDF_DIR = "context/context_docs/xpcs_publications"
OUTPUT  = "rag/citations/metadata_map.json"

def extract_doi(pdf_path):
    """Try to pull a DOI from PDF metadata fields and first 2 pages of text."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"    Could not open PDF: {e}")
        return None

    doi_pattern = r'10\.\d{4,}/[^\s\]>\"\'،،,]+'

    # 1. check all PDF metadata fields
    for val in doc.metadata.values():
        match = re.search(doi_pattern, str(val))
        if match:
            return match.group(0).strip(".")

    # 2. scan first 2 pages of text
    for i in range(min(2, len(doc))):
        text = doc[i].get_text()
        match = re.search(doi_pattern, text)
        if match:
            return match.group(0).strip(".")

    return None

def lookup_crossref(doi):
    """Fetch full citation metadata from CrossRef API."""
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": "xpcs-eval/1.0 (mailto:mmiller@anl.gov)"},
            timeout=10
        )
        if r.status_code == 200:
            d = r.json()["message"]
            authors = [
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in d.get("author", [])
            ]
            return {
                "title":   d.get("title",            [""])[0],
                "journal": d.get("container-title",  [""])[0],
                "year":    d.get("published", {}).get("date-parts", [[None]])[0][0],
                "authors": authors,
                "doi":     doi,
                "url":     f"https://doi.org/{doi}"
            }
        else:
            print(f"    CrossRef returned {r.status_code} for {doi}")
    except Exception as e:
        print(f"    CrossRef error for {doi}: {e}")
    return None

# --- main ---
pdfs = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])

metadata_map = {}
success      = 0
doi_only     = 0
failed       = 0

for filename in pdfs:
    path = os.path.join(PDF_DIR, filename)
    print(f"\n[{filename}]")

    doi = extract_doi(path)

    if doi:
        print(f"  DOI found: {doi}")
        meta = lookup_crossref(doi)
        if meta:
            metadata_map[filename] = meta
            print(f"  ✅ {meta['title'][:70]}")
            success += 1
        else:
            metadata_map[filename] = {"doi": doi, "url": f"https://doi.org/{doi}"}
            print(f"  ⚠️  CrossRef lookup failed — storing DOI only")
            doi_only += 1
        time.sleep(0.2)  # be polite to CrossRef
    else:
        metadata_map[filename] = {}
        print(f"  ❌ No DOI found")
        failed += 1

# save
with open(OUTPUT, "w") as f:
    json.dump(metadata_map, f, indent=2)

print(f"\n{'='*50}")
print(f"Total PDFs processed: {len(pdfs)}")
print(f"  ✅ Full metadata:   {success}")
print(f"  ⚠️  DOI only:       {doi_only}")
print(f"  ❌ No metadata:     {failed}")
print(f"\nSaved to {OUTPUT}")
