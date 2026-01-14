import PyPDF2
import os

def verify_citation(pdf_folder, filename, page_num, claim_snippet):
    """
    Quick manual verification helper
    """
    pdf_path = os.path.join(pdf_folder, filename)
    
    try:
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            # PDF pages are 0-indexed, but citations use 1-indexed
            page_text = pdf.pages[page_num].extract_text()
            
        print(f"\n{'='*80}")
        print(f"Checking: {filename} (Page {page_num})")
        print(f"Claim: {claim_snippet}")
        print(f"{'='*80}")
        print(f"\nPage text preview:\n{page_text[:500]}...")
        print(f"\n{'='*80}")
        
        # Simple keyword check
        keywords = claim_snippet.lower().split()[:5]  # First 5 words
        found = sum(1 for kw in keywords if kw in page_text.lower())
        
        print(f"✓ Found {found}/{len(keywords)} keywords")
        
        return found >= len(keywords) * 0.6  # 60% match threshold
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Test your citations
pdf_folder = "path/to/your/pdfs"

checks = [
    {
        "file": "003_Sutton_2008.pdf",
        "page": 4,
        "claim": "XPCS uses coherent X-ray scattering to study dynamics"
    },
    {
        "file": "006_Madsen_2016.pdf",
        "page": 7,
        "claim": "speckle contrast decreases with partial coherence"
    },
    {
        "file": "002_Gr_2008.pdf",
        "page": 5,
        "claim": "scattering volume comparable to coherence volume"
    }
]

for check in checks:
    verify_citation(pdf_folder, check['file'], check['page'], check['claim'])
    