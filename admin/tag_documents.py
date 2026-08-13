"""
Backfill tagger: assigns sample_type + freeform topics to every document in
the Qdrant collection using a lightweight LLM (gpt41nano by default). Writes
tags back to Qdrant via set_payload on all chunks of each source.

Usage:
  python -m admin.tag_documents                    # tag untagged docs only
  python -m admin.tag_documents --retag            # re-tag every doc
  python -m admin.tag_documents --source foo.pdf   # tag one specific doc
  python -m admin.tag_documents --limit 3          # cap docs tagged this run
  python -m admin.tag_documents --dry-run          # print, don't write
"""

import argparse
import json
import os
import re
import sys
import time

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from admin.weights_manager import get_all_docs
from admin.topic_taxonomy import SAMPLE_TYPES, build_tagging_prompt
from config import RERANKER_CONFIG

load_dotenv()

ARGO_API_URL = os.getenv("ARGO_API_URL", "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/")
ARGO_USER    = os.getenv("ARGO_USER", "")
COLLECTION   = os.getenv("QDRANT_COLLECTION_NAME", "xpcs_documents")

VALID_SAMPLE_TYPES = {s.lower() for s in SAMPLE_TYPES}


def _call_tagger(prompt: str) -> str:
    """Direct Argo call using RERANKER_CONFIG model (cheap/fast)."""
    payload = {
        "user": ARGO_USER,
        "model": RERANKER_CONFIG["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 500,
    }
    r = requests.post(ARGO_API_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "choices" in data:
        content = data["choices"][0]["message"]["content"]
    elif "response" in data:
        content = data["response"]
    elif "content" in data:
        content = data["content"]
    else:
        raise RuntimeError(f"Unexpected response shape: {data}")
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                          for part in content)
    return content


def _parse_tags(text: str) -> tuple[str, list[str]]:
    """Extract sample_type + topics from LLM output. Tolerant of extra prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return "", []
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "", []
    st_raw = str(obj.get("sample_type", "")).strip().lower()
    st = next((s for s in SAMPLE_TYPES if s.lower() == st_raw), "")
    topics_raw = obj.get("topics", [])
    if not isinstance(topics_raw, list):
        return st, []
    topics = []
    for t in topics_raw:
        t = str(t).strip().lower()
        if t and t not in topics:
            topics.append(t)
    return st, topics[:8]


def tag_document(doc: dict) -> tuple[str, list[str]]:
    authors = ", ".join(doc.get("authors", [])[:5]) or "unknown"
    prompt = build_tagging_prompt(
        title=doc.get("title") or doc["source"],
        authors=authors,
        excerpt=doc.get("text_excerpt", ""),
    )
    reply = _call_tagger(prompt)
    return _parse_tags(reply)


def write_tags(client: QdrantClient, source_full: str, sample_type: str, topics: list[str]) -> None:
    client.set_payload(
        collection_name=COLLECTION,
        payload={"sample_type": sample_type, "topics": topics},
        points=Filter(must=[FieldCondition(
            key="source", match=MatchValue(value=source_full)
        )]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retag", action="store_true", help="Re-tag docs that already have tags")
    ap.add_argument("--source", help="Tag only this source filename (basename)")
    ap.add_argument("--limit", type=int, help="Cap number of docs tagged this run")
    ap.add_argument("--dry-run", action="store_true", help="Print results, don't write to Qdrant")
    args = ap.parse_args()

    client = QdrantClient(host="localhost", port=6333)
    docs = get_all_docs(client, COLLECTION)

    if args.source:
        docs = [d for d in docs if d["source"] == args.source]
        if not docs:
            print(f"No document matches --source {args.source}")
            return
    if not args.retag:
        docs = [d for d in docs if not d.get("sample_type")]
    if args.limit:
        docs = docs[: args.limit]

    if not docs:
        print("Nothing to tag. (Use --retag to re-tag already-tagged docs.)")
        return

    print(f"Tagging {len(docs)} document(s) with {RERANKER_CONFIG['model']}"
          + (" [DRY RUN]" if args.dry_run else ""))
    print("─" * 70)

    ok, fail = 0, 0
    for i, doc in enumerate(docs, 1):
        print(f"[{i:>3}/{len(docs):>3}] {doc['source']}")
        try:
            sample_type, topics = tag_document(doc)
        except Exception as e:
            print(f"    ERROR: {e}")
            fail += 1
            continue

        if not sample_type:
            print("    WARN: LLM returned no valid sample_type — skipping write")
            fail += 1
            continue

        print(f"    sample_type: {sample_type}")
        print(f"    topics:      {', '.join(topics) if topics else '(none)'}")

        if not args.dry_run:
            try:
                write_tags(client, doc["source_full"], sample_type, topics)
                ok += 1
            except Exception as e:
                print(f"    QDRANT WRITE ERROR: {e}")
                fail += 1
        else:
            ok += 1

        time.sleep(0.15)

    print("─" * 70)
    print(f"Done. tagged={ok}  failed={fail}"
          + ("  (dry run — no writes)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
