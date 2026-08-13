"""
Backfill tagger: assigns sample_type + freeform topics to every document in
the Qdrant collection using a lightweight LLM (gpt41nano by default). Writes
tags back to Qdrant via set_payload on all chunks of each source.

Usage:
  python -m admin.tag_documents                    # tag untagged docs only
  python -m admin.tag_documents --retag            # re-tag every doc
  python -m admin.tag_documents --source foo.pdf   # tag one specific doc
  python -m admin.tag_documents --limit 3          # cap docs tagged this run
  python -m admin.tag_documents --dry-run          # print tags, don't write
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RERANKER_CONFIG  # noqa: E402
from admin.weights_manager import get_all_docs  # noqa: E402
from admin.topic_taxonomy import SAMPLE_TYPES, TAG_PROMPT_TMPL  # noqa: E402
import logger as applog  # noqa: E402

ARGO_API_URL = os.getenv("ARGO_API_URL", "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/")
ARGO_USER    = os.getenv("ARGO_USER", "")
COLLECTION   = os.getenv("QDRANT_COLLECTION_NAME", "xpcs_documents")


def _call_argo(model: str, prompt: str, max_tokens: int = 800) -> str:
    """Direct Argo call that adapts the max-tokens key per model family
    (mirrors pipeline.rerank_chunks). Returns "" on any failure."""
    if "gemini" in model:
        token_key = "max_output_tokens"
    elif model.startswith("gpt4") or model.startswith("gpt5") or model.startswith("o"):
        token_key = "max_completion_tokens"
    else:
        token_key = "max_tokens"
    payload = {
        "user":       ARGO_USER,
        "model":      model,
        "messages":   [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p":       1.0,
        token_key:     max_tokens,
    }
    resp = requests.post(ARGO_API_URL, json=payload, timeout=60)
    resp.raise_for_status()
    r = resp.json()
    if "choices" in r:
        content = r["choices"][0]["message"]["content"]
    elif "response" in r:
        content = r["response"]
    elif "content" in r:
        content = r["content"]
    else:
        return ""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content


def tag_document(doc: dict, model: str) -> tuple[str, list[str]]:
    """Ask the LLM to tag one document. Returns (sample_type, topics).
    Returns ("", []) on failure so the caller can skip cleanly."""
    prompt = TAG_PROMPT_TMPL.format(
        sample_types=", ".join(SAMPLE_TYPES),
        title=doc.get("title", "") or doc.get("source", ""),
        excerpt=(doc.get("text_excerpt") or "")[:2000],
    )
    try:
        raw = _call_argo(model, prompt)
    except requests.exceptions.RequestException as e:
        applog.log_api_network_error("tag_documents", e)
        return ("", [])
    except Exception as e:
        applog.log_error("tag_documents.call", str(e))
        return ("", [])
    if not isinstance(raw, str) or not raw.strip():
        return ("", [])

    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        applog.log_error("tag_documents.parse", f"no JSON in: {text[:200]}")
        return ("", [])
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError) as e:
        applog.log_error("tag_documents.json", f"{e}: {text[:200]}")
        return ("", [])

    raw_type = str(obj.get("sample_type", "")).strip().lower()
    lower_to_canonical = {s.lower(): s for s in SAMPLE_TYPES}
    if raw_type in lower_to_canonical:
        sample_type = lower_to_canonical[raw_type]
    else:
        applog.log_error("tag_documents.invalid_type", f"got {raw_type!r} for {doc.get('source')}")
        sample_type = "unclear"

    raw_topics = obj.get("topics", []) or []
    topics: list[str] = []
    seen = set()
    for t in raw_topics:
        if not isinstance(t, str):
            continue
        t = re.sub(r"\s+", " ", t.strip().lower())
        if t and t not in seen and len(t) <= 40:
            seen.add(t)
            topics.append(t)
    return (sample_type, topics[:8])


def write_tags(client: QdrantClient, source_full: str, sample_type: str, topics: list[str]) -> None:
    client.set_payload(
        collection_name=COLLECTION,
        payload={"sample_type": sample_type, "topics": topics},
        points=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_full))]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Tag Qdrant documents with sample_type + topics")
    ap.add_argument("--limit",   type=int, default=None, help="cap number of docs tagged this run")
    ap.add_argument("--retag",   action="store_true", help="re-tag docs that already have a sample_type")
    ap.add_argument("--source",  default=None, help="only tag the doc whose basename matches this")
    ap.add_argument("--dry-run", action="store_true", help="print tags without writing to Qdrant")
    args = ap.parse_args()

    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )
    docs = get_all_docs(client, COLLECTION)
    if args.source:
        docs = [d for d in docs if d["source"] == args.source]
    if not args.retag:
        docs = [d for d in docs if not d.get("sample_type")]
    if args.limit is not None:
        docs = docs[: args.limit]

    if not docs:
        print("[tag_documents] nothing to do (all docs already tagged, or filter matched none)")
        return

    model = RERANKER_CONFIG["model"]
    print(f"[tag_documents] tagging {len(docs)} document(s) with model={model} dry_run={args.dry_run}")

    tagged = 0
    for i, doc in enumerate(docs, 1):
        sample_type, topics = tag_document(doc, model)
        print(f"  [{i:>3}/{len(docs)}] {doc['source']}")
        print(f"      sample_type: {sample_type or '(FAILED)'}")
        print(f"      topics:      {', '.join(topics) or '(none)'}")
        if not sample_type:
            continue
        if args.dry_run:
            tagged += 1
            continue
        try:
            write_tags(client, doc["source_full"], sample_type, topics)
            tagged += 1
        except Exception as e:
            applog.log_error("tag_documents.write", f"{doc['source']}: {e}")
            print(f"      WRITE FAILED: {e}")

    print(f"[tag_documents] done — {tagged} tagged, {len(docs) - tagged} failed/skipped")


if __name__ == "__main__":
    main()
