# rag/citations/audit_metadata.py
import json

with open("rag/citations/metadata_map.json") as f:
    data = json.load(f)

print("=== MISSING METADATA (22) ===")
for filename, meta in sorted(data.items()):
    if not meta.get("title"):
        print(f"  {filename}")

print("\n=== DUPLICATE DOIs (potential wrong matches) ===")
from collections import Counter
doi_counts = Counter(
    meta.get("doi") for meta in data.values() if meta.get("doi")
)
for doi, count in doi_counts.items():
    if count > 1:
        files = [f for f, m in data.items() if m.get("doi") == doi]
        print(f"\n  DOI: {doi}  (appears {count}x)")
        for f in files:
            print(f"    {f}")
