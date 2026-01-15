# test_extraction.py - Enhanced version
from langchain_community.document_loaders import PyMuPDFLoader
import fitz  # PyMuPDF

pdf_path = "./context/context_docs/textbooks/x-ray-data-booklet.pdf"

print("Analyzing PDF structure...")
doc = fitz.open(pdf_path)

text_pages = 0
image_pages = 0
total_text_length = 0

for page_num in range(min(10, len(doc))):  # Check first 10 pages
    page = doc[page_num]
    text = page.get_text()
    images = page.get_images()
    
    total_text_length += len(text.strip())
    
    if len(text.strip()) > 50:
        text_pages += 1
    if images:
        image_pages += 1
    
    if page_num < 3:
        print(f"\nPage {page_num}:")
        print(f"  Text length: {len(text.strip())} chars")
        print(f"  Images: {len(images)}")
        print(f"  Text preview: {text.strip()[:100]}")

print("\n" + "="*80)
print(f"Total pages: {len(doc)}")
print(f"Pages with text (>50 chars): {text_pages}/10")
print(f"Pages with images: {image_pages}/10")
print(f"Average text per page: {total_text_length/10:.0f} chars")

if total_text_length < 500:
    print("\n⚠️  This appears to be a SCANNED/IMAGE-BASED PDF")
    print("    You'll need OCR to extract the content.")
else:
    print("\n✅ PDF has extractable text")
