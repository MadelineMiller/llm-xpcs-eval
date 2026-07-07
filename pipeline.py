"""
Shared pipeline functions used by app.py and the test suite.
No Chainlit dependency — safe to import anywhere.
"""

import os
import re
import requests
from dotenv import load_dotenv
from config import RETRIEVAL_CONFIG, RERANKER_CONFIG, LLM_CONFIG
import logger as applog

load_dotenv()

ARGO_API_URL = os.getenv('ARGO_API_URL', 'https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/')
ARGO_USER    = os.getenv('ARGO_USER', '')
COLLECTION   = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')


def display_page(page):
    """Convert 0-indexed DB page number to 1-indexed page number as printed in papers."""
    return (page + 1) if isinstance(page, int) else page


class _SyntheticPoint:
    """Wraps a Qdrant scroll Record so it has .id, .score, .payload like a query result."""
    def __init__(self, record, score):
        self.id      = record.id
        self.score   = score
        self.payload = record.payload


def extract_keywords(question: str) -> list:
    """Extract distinctive terms from the question for keyword-based retrieval."""
    stop_words = {
        'what', 'how', 'why', 'when', 'where', 'which', 'who', 'is', 'are',
        'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
        'that', 'this', 'these', 'those', 'and', 'or', 'but', 'not', 'from',
        'does', 'define', 'describe', 'explain', 'tell', 'give', 'about',
        'function', 'equation', 'formula', 'value', 'between', 'using',
    }
    acronyms = re.findall(r'\b[A-Z]{2,}\b', question)
    words = re.findall(r'[a-zA-Z]+', question.lower())
    long_words = [w for w in words if len(w) > 5 and w not in stop_words]
    long_words = sorted(set(long_words), key=len, reverse=True)[:2]
    return list(set(a.lower() for a in acronyms)) + long_words


def call_argo_llm(messages):
    """Call Argo API with conversation history."""
    payload = {
        "user": ARGO_USER,
        "model": LLM_CONFIG['model'],
        "messages": messages,
        "temperature": LLM_CONFIG['temperature'],
        "top_p": LLM_CONFIG['top_p'],
        "max_tokens": LLM_CONFIG['max_tokens'],
    }
    try:
        response = requests.post(ARGO_API_URL, json=payload, timeout=120)
        if not response.ok:
            applog.log_api_error("argo_llm", response.status_code, response.text)
            return f"API Error {response.status_code}: {response.text}"
        response.raise_for_status()
        result = response.json()
        if 'choices' in result:
            return result['choices'][0]['message']['content']
        elif 'response' in result:
            return result['response']
        elif 'content' in result:
            return result['content']
        else:
            return f"Unexpected response format: {result}"
    except requests.exceptions.RequestException as e:
        applog.log_api_network_error("argo_llm", e)
        return f"Network error calling Argo API: {str(e)}"
    except Exception as e:
        applog.log_error("argo_llm", str(e))
        return f"Error calling Argo API: {str(e)}"


def rerank_chunks(question: str, candidates: list, weights: dict = None) -> list:
    """Use LLM to filter retrieved chunks down to those that actually answer the question.

    weights: document weight dict (source filename -> 0-100). Higher-weight documents
    get a lower relevance threshold — include if possibly relevant, not just clearly relevant.
    """
    if not candidates:
        return []

    if weights is None:
        weights = {}

    cap     = RERANKER_CONFIG["max_candidates"]
    pool    = candidates[:cap]
    preview = RERANKER_CONFIG["preview_chars"]

    passages = []
    for i, pt in enumerate(pool):
        source = os.path.basename(pt.payload.get("source", ""))
        page = pt.payload.get("page", "?")
        w = weights.get(source, 50)
        text_preview = pt.payload.get("text", "")[:preview]
        passages.append(f"{i + 1}. [Priority: {w}/100] {text_preview}")
        print(f"[RERANKER] #{i+1:3d} | {source} p.{page} | {repr(text_preview[:80])}")
    passages_str = "\n\n".join(passages)

    prompt = (
        "You are a relevance filter for a scientific Q&A system about XPCS "
        "(X-ray Photon Correlation Spectroscopy).\n\n"
        "Question: " + question + "\n\n"
        "Below are " + str(len(pool)) + " passages from scientific literature, each tagged with "
        "a Priority score (0-100) set by the beamline scientist.\n\n"
        "Selection rules:\n"
        "- Priority ≥ 70: include if the passage is relevant OR possibly relevant to the question.\n"
        "- Priority 30–69: include only if clearly relevant.\n"
        "- Priority < 30: include only if directly and specifically relevant.\n\n"
        + passages_str + "\n\n"
        "Respond with ONLY a raw JSON object — no markdown, no code blocks, no explanation.\n"
        "Example: {\"relevant\": [1, 3, 5]}\n"
        "If none are relevant: {\"relevant\": []}"
    )

    model = RERANKER_CONFIG["model"]
    if "gemini" in model:
        token_key = "max_output_tokens"
    elif model.startswith("gpt4") or model.startswith("gpt5") or model.startswith("o"):
        token_key = "max_completion_tokens"
    else:
        token_key = "max_tokens"

    payload = {
        "user": ARGO_USER,
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1.0,
        token_key: RERANKER_CONFIG["max_tokens"],
    }

    try:
        response = requests.post(ARGO_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        if "choices" in result:
            text = result["choices"][0]["message"]["content"]
        elif "response" in result:
            text = result["response"]
        elif "content" in result:
            text = result["content"]
        else:
            return candidates

        text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text.strip())

        array_match = re.search(r'"relevant"\s*:\s*\[([^\]]*)', text, re.DOTALL)
        if array_match:
            indices = [int(n) for n in re.findall(r'\d+', array_match.group(1))]
            if indices:
                filtered = [pool[i - 1] for i in indices if 1 <= i <= len(pool)]
                print("[RERANKER] Kept", len(filtered), "of", len(pool), "chunks")
                return filtered
            else:
                print("[RERANKER] Model returned empty relevant list, using all candidates")
                applog.log_reranker_empty(text)
                return pool

        print("[RERANKER] Could not parse response, using all candidates:", text[:100])
        applog.log_reranker_fallback(text)
    except Exception as e:
        print("[RERANKER] Error:", e, "— falling back to all candidates")
        applog.log_reranker_error(e)

    return pool
