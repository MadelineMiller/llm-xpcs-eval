# rag/citations/merge_patches.py
import json
from collections import Counter

MAP_FILE     = "rag/citations/metadata_map.json"
PATCHES_FILE = "rag/citations/manual_patches.json"


with open(MAP_FILE) as f:
    metadata_map = json.load(f)


with open(PATCHES_FILE) as f:
    patches = json.load(f)



for filename, patch in patches.items():
    if filename not in metadata_map:
        print(f"  ⚠️  {filename} not in metadata_map — adding anyway")
    metadata_map[filename] = patch
    action = "SKIP" if patch.get("SKIP") else "patched"
    print(f"  {action}: {filename}")



with open(MAP_FILE, "w") as f:
    json.dump(metadata_map, f, indent=2)


print(f"\n{'='*50}")
print(f"Total entries: {len(metadata_map)}")



full     = sum(1 for v in metadata_map.values() if v.get("title") and not v.get("SKIP"))
no_doi   = sum(1 for v in metadata_map.values() if v.get("title") and not v.get("doi") and not v.get("SKIP"))
skipped  = sum(1 for v in metadata_map.values() if v.get("SKIP"))
empty    = sum(1 for v in metadata_map.values() if not v.get("title") and not v.get("SKIP"))


print(f"  ✅ Has title + DOI:   {full - no_doi}")
print(f"  📖 Has title, no DOI: {no_doi}  (textbooks — expected)")
print(f"  ⏭️  Skipped:           {skipped}")
print(f"  ❌ Still empty:       {empty}")


if empty > 0:
    print(f"\nStill empty:")
    for k, v in sorted(metadata_map.items()):
        if not v.get("title") and not v.get("SKIP"):
            print(f"  {k}")

print(f"\n=== Duplicate DOI check ===")
doi_counts = Counter(
    v.get("doi") for v in metadata_map.values()
    if v.get("doi") and not v.get("SKIP")
)
dupes = {doi: count for doi, count in doi_counts.items() if count > 1}
if dupes:
    for doi, count in dupes.items():
        files = [f for f, v in metadata_map.items() if v.get("doi") == doi]
        print(f"  {doi} ({count}x): {files}")
else:
    print("  No duplicates ✅")