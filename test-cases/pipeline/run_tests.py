"""
End-to-end pipeline tests for the XPCS chatbot.

Runs the full retrieval + reranker + LLM pipeline without the Chainlit UI
and checks that the LLM answer contains expected content from specific paper pages.

Run with:
    python3 test-cases/pipeline/run_tests.py
"""

import argparse
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText, MatchValue

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RETRIEVAL_CONFIG
from admin.weights_manager import load_weights, apply_weights
from pipeline import (
    extract_keywords, rerank_chunks, call_argo_llm,
    _SyntheticPoint, display_page, COLLECTION,
)


# ── Test cases ────────────────────────────────────────────────────────────────
# Each test case specifies:
#   question              : the user's question
#   expected_source       : substring to identify the target paper (matched against source path or title)
#   expected_page         : 1-indexed page the answer should come from
#   expected_chunk_contains : substring that must appear in the retrieved chunk text
#   expected_answer_contains: list of substrings that must ALL appear in the LLM's answer

TEST_CASES = [
    {
        "id": "aps_beamline_photon_energy",
        "question": (
            "What photon energy was used to do X-ray diffraction measurements that were "
            "performed at APS at beamline 6-ID-D using a 2D amorphous silicon area detector?"
        ),
        "expected_source": "amorphous ice",
        "expected_page": 5,
        "expected_chunk_contains": "100 kev photon energy",
        "expected_answer_contains": [
            {
                "label": "title cited in LLM response",
                "text": "diffusive dynamics during the high-to-low density transition in amorphous ice",
            },
            {
                "label": "correct page cited (p.5)",
                "text": "page 5",
            },
            {
                "label": "correct photon energy (100 keV)",
                "text": "100 kev",
            },
            {
                "label": "correct beamline (6-ID-D)",
                "text": "6-id-d",
            },
        ],
    },
    {
        "id": "chi_t_normalized_variance",
        "question": "What is the normalized variance χT of the temporal autocorrelation function?",
        "expected_source": "amorphous ice",
        "expected_page": 5,
        "expected_chunk_contains": (
            "quantitative measure of the dynamical heterogeneities"
        ),
        "expected_answer_contains": [
            {
                "label": "title cited in LLM response",
                "text": "diffusive dynamics during the high-to-low density transition in amorphous ice",
            },
            {
                "label": "correct page cited (p.5)",
                "text": "page 5",
            },
            {
                "label": "definition of χT used in answer",
                "text": "quantitative measure of the dynamical heterogeneities",
            },
            {
                "label": "formula provided",
                "text": "= 1/n [",
            },
        ],
    },
]


# ── Pipeline ──────────────────────────────────────────────────────────────────
# NOTE: retrieve() below mirrors the retrieval logic in app.py (search for
# "Primary: broad semantic search"). If that logic changes, update this too.

def retrieve(question: str, embeddings, client) -> list:
    query_vector = embeddings.embed_query(question)

    base_filter = {
        "must_not": [{"key": "source", "match": {"text": "x-ray-data-booklet"}}]
    }

    # Primary: vector search
    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=RETRIEVAL_CONFIG["num_results"],
        query_filter=base_filter,
    )
    combined = list(results.points)
    seen_ids = {p.id for p in combined}

    # Secondary: keyword scroll
    keywords = extract_keywords(question)
    if keywords:
        kw_filter = Filter(
            must_not=[FieldCondition(key="source", match=MatchValue(value="x-ray-data-booklet"))],
            must=[FieldCondition(key="text", match=MatchText(text=kw)) for kw in keywords],
        )
        kw_records, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=kw_filter,
            limit=30,
            with_payload=True,
            with_vectors=False,
        )
        for rec in kw_records:
            if rec.id not in seen_ids:
                combined.append(_SyntheticPoint(rec, score=0.9))
                seen_ids.add(rec.id)

    # Tertiary: adjacent chunks — score inherits from the parent doc's best score
    doc_pages = {}
    doc_best_score = {}
    for p in combined:
        src  = p.payload.get("source", "")
        page = p.payload.get("page")
        if src and page is not None:
            doc_pages.setdefault(src, set()).add(int(page))
            doc_best_score[src] = max(doc_best_score.get(src, 0.0), p.score)

    for src, pages in doc_pages.items():
        adjacent = set()
        for page in pages:
            if page > 0:
                adjacent.add(page - 1)
            adjacent.add(page + 1)
        adjacent -= pages
        if not adjacent:
            continue
        adj_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=src))],
            should=[FieldCondition(key="page", match=MatchValue(value=p)) for p in adjacent],
        )
        adj_records, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=adj_filter,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        for rec in adj_records:
            if rec.id not in seen_ids:
                parent_score = doc_best_score.get(src, 0.80)
                combined.append(_SyntheticPoint(rec, score=parent_score * 0.95))
                seen_ids.add(rec.id)

    return combined


def generate_answer(question: str, chunks: list) -> str:
    if not chunks:
        return "No relevant passages found."

    context = "\n\n".join(
        "[" + (c.payload.get("title") or os.path.basename(c.payload.get("source", "")))
        + ", Page " + str(display_page(c.payload.get("page", "?"))) + "]\n"
        + c.payload.get("text", "")
        for c in chunks
    )
    prompt = (
        "Answer the question below using ONLY the provided passages. "
        "Quote relevant definitions or formulas directly from the text. "
        "Cite sources inline by title and page.\n\n"
        "CONTEXT:\n" + context + "\n\n"
        "QUESTION: " + question
    )
    return call_argo_llm([{"role": "user", "content": prompt}])


# ── Runner ────────────────────────────────────────────────────────────────────

def run_test(tc: dict, embeddings, client, verbose: bool = False) -> bool:
    print("\n" + "=" * 70)
    print("TEST:", tc["id"])
    print("  Q:", tc["question"])

    # Retrieval — suppress chatty per-chunk logging unless --verbose
    sink = sys.stdout if verbose else StringIO()
    with redirect_stdout(sink):
        candidates = retrieve(tc["question"], embeddings, client)
        weights    = load_weights()
        candidates = apply_weights(candidates, weights)

    # Rerank
    with redirect_stdout(sink):
        kept = rerank_chunks(tc["question"], candidates, weights)
    kept = kept[:100]
    print(f"  Retrieved {len(candidates)} candidates → reranker kept {len(kept)}")

    # Check that the expected chunk was retrieved
    src_key   = tc["expected_source"].lower()
    page_key  = tc["expected_page"]
    chunk_key = tc["expected_chunk_contains"].lower()

    target_chunk = None
    for c in kept:
        src_match  = src_key in (c.payload.get("source", "") + c.payload.get("title", "")).lower()
        page_match = display_page(c.payload.get("page")) == page_key
        text_match = chunk_key in c.payload.get("text", "").lower()
        if src_match and page_match and text_match:
            target_chunk = c
            break

    print(f"  Chunk checks (source='{tc['expected_source']}', p.{page_key}):")
    chunk_text_pass = target_chunk is not None
    print(f"    {'PASS' if chunk_text_pass else 'FAIL'} — chunk contains: \"{tc['expected_chunk_contains'][:80]}...\"")
    if not chunk_text_pass:
        chunks_from_source = [
            c for c in kept
            if src_key in (c.payload.get("source", "") + c.payload.get("title", "")).lower()
        ]
        if chunks_from_source:
            print("    Chunks from that source that were kept:")
            for c in chunks_from_source:
                print("      p." + str(display_page(c.payload.get("page"))) + " — "
                      + c.payload.get("text", "")[:100].replace("\n", " ") + "...")
        else:
            print(f"    No chunks from '{tc['expected_source']}' made it through reranker")

    # Generate answer and check each expected substring
    print("  Generating answer...")
    llm_answer = generate_answer(tc["question"], kept)
    answer_lower = llm_answer.lower()

    check_results = [(c["label"], c["text"], c["text"].lower() in answer_lower)
                     for c in tc["expected_answer_contains"]]
    answer_pass = all(ok for _, _, ok in check_results)

    print("  Answer checks:")
    for label, text, ok in check_results:
        print(f"    {'PASS' if ok else 'FAIL'} — {label}: \"{text[:70]}\"")

    print("\n  LLM Answer:\n")
    for line in llm_answer.splitlines():
        print("    " + line)

    passed = target_chunk is not None and answer_pass
    print("\n  OVERALL: " + ("PASS" if passed else "FAIL"))
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-chunk weight and reranker debug output")
    args = parser.parse_args()

    print("Loading embeddings (this takes ~30s)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="allenai/scibert_scivocab_uncased",
        model_kwargs={"device": "cpu"},
    )
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    results = [run_test(tc, embeddings, client, verbose=args.verbose) for tc in TEST_CASES]

    passed = sum(results)
    failed = len(results) - passed
    print("\n" + "=" * 70)
    print("RESULTS:", passed, "passed,", failed, "failed out of", len(TEST_CASES), "tests")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
