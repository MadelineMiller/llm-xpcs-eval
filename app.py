from admin.admin import launch_admin
from admin.weights_manager import load_weights, save_weights, get_all_docs, apply_weights
launch_admin()

import asyncio
import functools
import os
import json
import requests

import chainlit as cl
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from config import RETRIEVAL_CONFIG, LLM_CONFIG, RERANKER_CONFIG

import ldap

import secrets
from auth_tokens import admin_auth_tokens, add_token, remove_token
import logger as applog


# ============================================================================
# DEMO HELPER FUNCTIONS
# ============================================================================

import re


_TITLE_LOWERCASE = {
    'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'so', 'yet',
    'at', 'by', 'in', 'of', 'on', 'to', 'up', 'as', 'via', 'vs',
}

def clean_title(title):
    """Strip MathML/XML tags and apply title case to CrossRef titles."""
    if not title:
        return title
    title = re.sub(r'<[^>]+>', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    # Apply title case, keeping small words lowercase except at the start
    words = title.split()
    cased = [
        word if (i > 0 and word.lower() in _TITLE_LOWERCASE) else word.capitalize()
        for i, word in enumerate(words)
    ]
    return ' '.join(cased)

def build_source_elements(results):
    """Build one side-panel Text element per chunk, named 'Title, p.N'.
    The LLM cites as 'Source: [Title, p.N]' so each citation opens exactly that chunk."""
    elements = []
    seen_names = set()
    source_chunks = {}

    for result in results:
        p = result.payload
        title = clean_title(p.get("title")) or os.path.basename(p["source"])
        raw_page = p.get("page", "?")
        display_page = (raw_page + 1) if isinstance(raw_page, int) else raw_page
        name = f"{title}, p.{display_page}"

        if name in seen_names:
            continue
        seen_names.add(name)

        authors = p.get("authors") or []
        journal = p.get("journal") or ""
        year    = p.get("year") or ""
        doi     = p.get("doi") or ""
        url     = p.get("url") or ""

        if len(authors) > 2:
            author_str = f"{authors[0]} et al."
        elif authors:
            author_str = ", ".join(authors)
        else:
            author_str = ""

        meta_lines = []
        if author_str:
            meta_lines.append(f"**Authors:** {author_str}")
        if journal:
            meta_lines.append(f"**Journal:** {journal}")
        meta_lines.append(f"**Page:** {display_page}")
        if year:
            meta_lines.append(f"**Year:** {year}")
        if doi and url:
            meta_lines.append(f"**DOI:** [{doi}]({url})")
        elif doi:
            meta_lines.append(f"**DOI:** {doi}")

        chunk_text = p.get("text", "")
        side_content = (
            "\n\n".join(meta_lines)
            + "\n\n<br>\n\n"
            + f"<details><summary>📄 Show chunk text (p.{display_page})</summary>\n\n"
            + "*Extracted directly from the PDF — formatting may be inconsistent.*\n\n"
            + "---\n\n"
            + chunk_text
            + "\n\n</details>"
        )

        elements.append(cl.Text(name=name, content=side_content, display="side"))
        source_chunks[name] = [p]

    return elements, [], source_chunks


# ============================================================================
# INITIALIZATION
# ============================================================================

load_dotenv()

# ldap

LDAP_SERVER = os.getenv("LDAP_SERVER", "")
LDAP_BASE_DN = os.getenv("LDAP_BASE_DN", "")
LDAP_SERVICE_USER_DN = os.getenv("LDAP_SERVICE_USER_DN", "")
LDAP_ADMIN_PASSWORD = os.getenv("LDAP_ADMIN_PASSWORD", "")

# admin ranking system page
APP_HOST = os.getenv("APP_HOST", "http://localhost")

print("Initializing XPCS Hypothesis Evaluator...")
embeddings = HuggingFaceEmbeddings(
    model_name="allenai/scibert_scivocab_uncased",
    model_kwargs={'device': 'cpu'}
)

client = QdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 6333))
)

ARGO_API_URL = os.getenv('ARGO_API_URL', 'https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/')
ARGO_USER = os.getenv('ARGO_USER', 'your_anl_username')

print("Ready!")


# ============================================================================
# API FUNCTIONS
# ============================================================================

def call_argo_llm(messages):
    """Call Argo API with conversation history."""

    payload = {
        "user": ARGO_USER,
        "model": LLM_CONFIG['model'],
        "messages": messages,
        "temperature": LLM_CONFIG['temperature'],
        "top_p": LLM_CONFIG['top_p'],
        "max_tokens": LLM_CONFIG['max_tokens']
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


def extract_keywords(question: str) -> list:
    """Extract distinctive terms from the question for keyword-based retrieval."""
    stop_words = {
        'what', 'how', 'why', 'when', 'where', 'which', 'who', 'is', 'are',
        'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
        'that', 'this', 'these', 'those', 'and', 'or', 'but', 'not', 'from',
        'does', 'define', 'describe', 'explain', 'tell', 'give', 'about',
        'function', 'equation', 'formula', 'value', 'between', 'using',
    }
    # Acronyms: all-caps words of any length (XIFS, XPCS, SAXS, etc.)
    acronyms = re.findall(r'\b[A-Z]{2,}\b', question)

    # Long distinctive words (> 5 chars, not stop words)
    words = re.findall(r'[a-zA-Z]+', question.lower())
    long_words = [w for w in words if len(w) > 5 and w not in stop_words]
    long_words = sorted(set(long_words), key=len, reverse=True)[:2]

    return list(set(a.lower() for a in acronyms)) + long_words


def rerank_chunks(question: str, candidates: list, weights: dict = None) -> list:
    """Use LLM to filter retrieved chunks down to those that actually answer the question.

    weights: document weight dict (source filename -> 0-100). Higher-weight documents
    get a lower relevance threshold — include if possibly relevant, not just clearly relevant.
    """
    if not candidates:
        return []

    if weights is None:
        weights = {}

    # Cap candidates to keep the reranker prompt small and fast
    cap = RERANKER_CONFIG["max_candidates"]
    pool = candidates[:cap]
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

        # Strip markdown code fences if the model wrapped the JSON anyway
        text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text.strip())

        # Extract indices — works even on truncated JSON (no closing ]} needed)
        array_match = re.search(r'"relevant"\s*:\s*\[([^\]]*)', text, re.DOTALL)
        if array_match:
            indices = [int(n) for n in re.findall(r'\d+', array_match.group(1))]
            if indices:
                filtered = [pool[i - 1] for i in indices if 1 <= i <= len(pool)]
                print("[RERANKER] Kept", len(filtered), "of", len(pool), "chunks")
                return filtered
            else:
                # Valid empty response — reranker found nothing relevant; fall back to all
                print("[RERANKER] Model returned empty relevant list, using all candidates")
                applog.log_reranker_empty(text)
                return pool

        print("[RERANKER] Could not parse response, using all candidates:", text[:100])
        applog.log_reranker_fallback(text)
    except Exception as e:
        print("[RERANKER] Error:", e, "— falling back to all candidates")
        applog.log_reranker_error(e)

    return pool


# ============================================================================
# CHAINLIT HANDLERS
# ============================================================================

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if not LDAP_ADMIN_PASSWORD:
        print("[AUTH] LDAP_ADMIN_PASSWORD not set in .env")
        return None
    
    # Sanitize username to prevent LDAP injection
    if not username.isalnum():
        print(f"[AUTH] Invalid username format: {username}")
        applog.log_login_failure(username, "invalid_username_format")
        return None

    try:
        ldap.set_option(ldap.OPT_X_TLS_REQUIRE_CERT, ldap.OPT_X_TLS_NEVER)
        conn = ldap.initialize(LDAP_SERVER)
        conn.set_option(ldap.OPT_REFERRALS, 0)
        conn.set_option(ldap.OPT_PROTOCOL_VERSION, ldap.VERSION3)

        conn.simple_bind_s(LDAP_SERVICE_USER_DN, LDAP_ADMIN_PASSWORD)

        search_filter = f"(&(cn={username}))"
        result = conn.search_s(LDAP_BASE_DN, ldap.SCOPE_SUBTREE, search_filter)

        user_dn, user_info = result[0]
        if not user_dn:
            print(f"[AUTH] User not found: {username}")
            applog.log_login_failure(username, "user_not_found")
            return None

        try:
            conn.simple_bind_s(user_dn, password)
        except ldap.INVALID_CREDENTIALS:
            print(f"[AUTH] Invalid password for: {username}")
            applog.log_login_failure(username, "invalid_credentials")
            return None

        first_name = user_info.get("givenName", [b""])[0].decode()
        last_name = user_info.get("sn", [b""])[0].decode()
        email = user_info.get("mail", [b""])[0].decode()

        print(f"[AUTH] Login successful: {username} ({first_name} {last_name})")
        applog.log_login_success(username, f"{first_name} {last_name}", email)

        return cl.User(
            identifier=username,
            metadata={"name": f"{first_name} {last_name}", "email": email}
        )

    except Exception as e:
        print(f"[AUTH ERROR] {e}")
        applog.log_error("AUTH_LDAP", str(e))
        return None


@cl.on_chat_start
async def start():
    user = cl.user_session.get("user")
    username = getattr(user, "identifier", "unknown") if user else "unknown"
    applog.log_session_start(username)

    # Generate admin access token for this session
    token = secrets.token_urlsafe(32)
    add_token(token)
    cl.user_session.set("admin_token", token)

    cl.user_session.set("conversation_history", [])
    
    # UPDATED SYSTEM PROMPT - More explicit instructions
    system_prompt = {
        "role": "system",
        "content": """You are Argo, an expert AI assistant for X-ray Photon Correlation Spectroscopy (XPCS) at Argonne National Laboratory's Advanced Photon Source.

**CRITICAL INSTRUCTION - READ CAREFULLY:**

When you receive context passages from XPCS literature, you MUST use them to construct your answer. Do NOT claim "the literature doesn't provide information" unless the passages are truly irrelevant or off-topic.

**How to Use Context Passages:**

1. **Synthesize Information:**
   - If passages discuss speckle patterns → that IS information about XPCS
   - If passages discuss coherence and dynamics → that IS information about XPCS
   - If passages contain formulas or experimental details → USE THEM
   - Combine information from multiple passages to build a complete answer

2. **Citation Requirements:**
   - After each sentence or claim, write the citation on its own new line in this exact format:
     Source: [Exact Paper Title, p.N]
     where N is the page number from the context. Example:
     "XPCS measures dynamics via speckle fluctuations.
     Source: [X-Ray Photon Correlation Spectroscopy, p.5]"
   - Use the exact title as it appears in the context — do not shorten or paraphrase it.
   - Include formulas exactly as written: $g^{(2)}(q,t) = \langle I(q,0)I(q,t) \rangle / \langle I(q) \rangle^2$
   - Quote key sentences when appropriate

3. **When to Acknowledge Missing Information:**
   ONLY say "information not found" when:
   - Passages are about completely different topics (e.g., asking about XPCS but getting passages about protein crystallography)
   - Question asks for specific beamline specs (flux, energy, detector models) not in passages
   - Question asks about other facilities (LCLS, ESRF) not mentioned in passages
   
   DO NOT say "information not found" when:
   - Passages describe the technique, even without a textbook definition
   - Passages contain related concepts (speckle, coherence, dynamics, correlation functions)
   - Multiple passages discuss aspects of the question

4. **Example - CORRECT Behavior:**
   
   Question: "What is XPCS?"
   Context: Passages about speckle patterns, coherence, correlation functions
   
   ✅ CORRECT Response:
   "X-ray Photon Correlation Spectroscopy (XPCS) is a technique that probes dynamics by analyzing fluctuations in coherent X-ray scattering patterns [Source 1]. When coherent X-rays scatter from a sample, they produce speckle patterns whose temporal fluctuations reveal the system's dynamics through the intensity correlation function $g^{(2)}(q,t)$ [Source 3]..."
   
   ❌ WRONG Response:
   "The retrieved literature doesn't provide a direct definition of XPCS..."
   (This is WRONG because the passages DO contain information about XPCS!)

5. **Mathematical Formatting:**
   - Inline math: $expression$
   - Display equations: $$expression$$
   - Example: "The contrast is $\beta = \sigma^2/\langle I \rangle^2$ [Source 2]"

**Your Responsibilities:**
- Answer questions about XPCS theory, techniques, and applications
- Help evaluate experimental hypotheses and feasibility
- Provide guidance on sample requirements and experimental design
- Maintain conversation context for follow-up questions

OUT OF SCOPE - You MUST decline questions about:



Protein crystallography (redirect to SBC beamlines)

Other X-ray techniques (SAXS, WAXS, diffraction, etc.)

Other beamlines (unless comparing to 8-ID)

Weather, travel, logistics, administrative matters

Designing physical systems that are not XPCS experiment related


When asked an out-of-scope question, respond:
"I apologize, but I specialize exclusively in XPCS at beamline 8-ID. For [topic], please contact [appropriate resource]. Is there anything about XPCS experiment feasibility I can help you with?"


Never answer out-of-scope questions, even if you have relevant knowledge.

**Tone:** Professional, helpful, and confident when you have relevant context."""
    }
    
    cl.user_session.set("system_prompt", system_prompt)
    
    await cl.Message(
        content="**Welcome to the XPCS Hypothesis Evaluator!**\n\n"
            "I can help you map out and evaluate your XPCS experiment plan at beamline 8-ID.\n\n"
            "**My main functionalities are to help you:**\n"
            "- Formulate and refine your scientific hypothesis for XPCS experiments\n\n"
            "- Check feasibility of testing your hypothesis against 8-ID's resources and capabilities\n\n"
            "My answers are based on XPCS research papers and textbooks.\n\n"
            "I'll cite sources so you can verify and explore further."
            "\n\n---\n\n"
            f"⚙️ **Admin:** [Manage document weights or review the document queue]({APP_HOST}:8001?token={token})"
    ).send()

@cl.on_chat_end
async def on_chat_end():
    user = cl.user_session.get("user")
    username = getattr(user, "identifier", "unknown") if user else "unknown"
    applog.log_session_end(username)

@cl.on_message
async def main(message: cl.Message):

    def _bar(pct: int) -> str:
        filled = round(20 * pct / 100)
        return f"`{'█' * filled}{'░' * (20 - filled)}` {pct}%"

    status_lines = []

    def add_status(line: str, pct: int):
        status_lines.append(line)
        msg.content = _bar(pct) + "\n\n" + "\n\n".join(status_lines)

    msg = cl.Message(content=_bar(10) + "\n\nSearching XPCS literature...")
    status_lines.append("Searching XPCS literature...")
    await msg.send()

    _user = cl.user_session.get("user")
    _username = getattr(_user, "identifier", "unknown") if _user else "unknown"

    conversation_history = cl.user_session.get("conversation_history")
    system_prompt = cl.user_session.get("system_prompt")

    def expand_query(query):
        """Expand query with related terms for better retrieval"""
        expansions = {
            "what is xpcs": "XPCS X-ray Photon Correlation Spectroscopy speckle dynamics correlation",
            "speckle contrast": "speckle contrast beta variance intensity fluctuations",
            "correlation function": "correlation function g2 intensity autocorrelation dynamics",
        }
        
        query_lower = query.lower()
        for key, expansion in expansions.items():
            if key in query_lower: 
                return f"{query} {expansion}"
        
        return query

    expanded_query = expand_query(message.content)
    
    # Search for relevant context
    query_vector = embeddings.embed_query(expanded_query)

    base_filter = {
        "must_not": [{"key": "source", "match": {"text": "x-ray-data-booklet"}}]
    }

    # Primary: broad semantic search
    results = client.query_points(
        collection_name=os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents'),
        query=query_vector,
        limit=RETRIEVAL_CONFIG['num_results'],
        query_filter=base_filter,
    )
    combined_points = list(results.points)
    seen_ids = {p.id for p in combined_points}
    vec_count = len(combined_points)

    add_status("Searching by keyword...", pct=30)
    await msg.update()

    from qdrant_client.models import Filter, FieldCondition, MatchText, MatchValue

    class _SyntheticPoint:
        """Wraps a scroll Record to be compatible with apply_weights (.id, .score, .payload)."""
        def __init__(self, record, score):
            self.id = record.id
            self.score = score
            self.payload = record.payload

    COLLECTION = os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents')

    # Secondary: keyword scroll — finds chunks containing ALL key terms regardless of vector rank
    keywords = extract_keywords(message.content)
    kw_added = 0
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
        for record in kw_records:
            if record.id not in seen_ids:
                combined_points.append(_SyntheticPoint(record, score=0.9))
                seen_ids.add(record.id)
                kw_added += 1

    add_status("Collecting surrounding context from relevant papers...", pct=55)
    await msg.update()

    # Tertiary: adjacent chunk retrieval — for each retrieved doc, also fetch neighboring pages
    # so that definitions/context on adjacent pages aren't missed
    doc_pages = {}
    doc_best_score = {}  # src → highest score among its retrieved chunks
    for point in combined_points:
        src = point.payload.get('source', '')
        page = point.payload.get('page')
        if src and page is not None:
            doc_pages.setdefault(src, set()).add(int(page))
            doc_best_score[src] = max(doc_best_score.get(src, 0.0), point.score)

    adj_added = 0
    print(f"[RETRIEVE] Adjacent fetch: {len(doc_pages)} docs, {len(combined_points)} candidates so far")
    for src, pages in doc_pages.items():
        adjacent_pages = set()
        for page in pages:
            if page > 0:
                adjacent_pages.add(page - 1)
            adjacent_pages.add(page + 1)
        adjacent_pages -= pages  # don't re-fetch already-retrieved pages

        if not adjacent_pages:
            continue

        adj_filter = Filter(
            must=[
                FieldCondition(key="source", match=MatchValue(value=src)),
            ],
            should=[
                FieldCondition(key="page", match=MatchValue(value=p))
                for p in adjacent_pages
            ],
        )
        adj_records, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=adj_filter,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        for record in adj_records:
            if record.id not in seen_ids:
                parent_score = doc_best_score.get(src, 0.80)
                combined_points.append(_SyntheticPoint(record, score=parent_score * 0.95))
                seen_ids.add(record.id)
                adj_added += 1

    print("Keywords:", keywords, "| kw added:", kw_added, "| adjacent added:", adj_added)

    total = len(combined_points)
    add_status("Checking which passages are most relevant to your question...", pct=75)
    await msg.update()

    # Wrap as a simple namespace so apply_weights works unchanged
    class _Results:
        def __init__(self, points):
            self.points = points

    results = _Results(combined_points)

    current_weights = load_weights()

    # Apply weights and re-sort by priority
    reranked_points = apply_weights(results.points, current_weights)

    print("\nQuery:", message.content)
    print("Retrieved", len(reranked_points), "candidates from Qdrant")
    for idx, result in enumerate(reranked_points, 1):
        source = os.path.basename(result.payload['source'])
        print("  [" + str(idx) + "] score=" + str(round(result.score, 4)) + " | " + source)

    # LLM reranker: filter candidates to those that actually answer the question
    # Run in executor so the async event loop (and WebSocket keep-alive) stays alive
    reranked_points = await asyncio.get_event_loop().run_in_executor(
        None, functools.partial(rerank_chunks, message.content, reranked_points, current_weights)
    )
    reranked_points = reranked_points[:100]

    kept = len(reranked_points)
    add_status("Generating answer...", pct=90)
    await msg.update()

    context_parts = []
    sources = []
    seen_titles = set()
    cite_key_map = {}  # SRC-N → exact element name

    for idx, result in enumerate(reranked_points, 1):
        p = result.payload
        source = os.path.basename(p['source'])
        page = p['page']
        text = p['text']
        title = clean_title(p.get('title')) or source
        weight = current_weights.get(source, 50)

        display_page = (page + 1) if isinstance(page, int) else page
        element_name = f"{title}, p.{display_page}"
        src_key = f"SRC-{idx}"
        cite_key_map[src_key] = element_name
        context_parts.append(
            f"[CITE: {src_key} | Priority: {weight}/100]\n{text}"
        )
        sources.append(f"[{idx}] {element_name}")
        seen_titles.add(title)



    # Build context message for LLM
    if context_parts:
        context = "\n\n".join(context_parts)
        context_message = f"""CONTEXT FROM XPCS SCIENTIFIC LITERATURE:

(Passages are ordered by source priority — earlier sources are from higher-priority documents.)

{context}

---

USER QUESTION: {message.content}

---

INSTRUCTIONS FOR YOUR RESPONSE:

1. The passages above ARE relevant to the question - use them!
2. Prioritize information from earlier sources — they are from higher-priority documents as rated by the beamline scientist
3. When sources conflict, prefer the higher-priority (earlier) source
4. Build your answer by synthesizing information from these passages
5. After each sentence or claim that uses a source, write on its own new line:
   Source: [CITE key]
   The CITE key is the short code in the passage header (e.g. SRC-1, SRC-2).
   Example — if the header says [CITE: SRC-3 | Priority: ...], write:
   Source: [SRC-3]
6. Include any formulas or technical details from the passages
7. If passages discuss related concepts (speckle, coherence, dynamics), explain how they relate to the question
8. Use LaTeX for math: $inline$ or $$display$$

DO NOT say "the literature doesn't provide information" - you have {len(context_parts)} relevant passages above!

Provide a comprehensive, well-cited answer based on the passages."""
        
    else:
        context_message = (
            "No relevant passages were found for this question.\n\n"
            "USER QUESTION: " + message.content + "\n\n"
            "Since no relevant passages were retrieved, respond with:\n"
            "\"I don't have specific information about this in the XPCS literature database. "
            "For [topic], please consult [appropriate resource].\"\n\n"
            "Do NOT attempt to answer from general knowledge."
        )
    
    # Build messages for Argo API
    messages = [system_prompt]
    messages.extend(conversation_history[-10:])
    messages.append({
        "role": "user",
        "content": context_message
    })
    
    # Call Argo LLM — run in executor so the WebSocket keep-alive isn't blocked
    answer = await asyncio.get_event_loop().run_in_executor(
        None, functools.partial(call_argo_llm, messages)
    )

    # Expand SRC-N keys to exact element names so Chainlit can match them
    def _expand_cite(m):
        prefix, key = m.group(1), m.group(2).strip()
        return prefix + cite_key_map.get(key, key)

    answer = re.sub(r'^(Source:\s*)\[([^\]]+)\]\s*$', _expand_cite, answer, flags=re.MULTILINE)

    # Log the full query interaction
    applog.log_query(
        username=_username,
        question=message.content,
        keywords=keywords,
        vec_count=vec_count,
        kw_added=kw_added,
        adj_added=adj_added,
        kept=kept,
        sources=[
            f"{p.payload.get('title') or os.path.basename(p.payload.get('source', ''))} (page {p.payload.get('page', '?')})"
            for p in reranked_points
        ],
        context="\n\n".join(context_parts),
        answer=answer,
    )

    # Update conversation history
    conversation_history.append({"role": "user", "content": message.content})
    conversation_history.append({"role": "assistant", "content": answer})
    cl.user_session.set("conversation_history", conversation_history)
    
    # Build source side-panel elements
    acknowledged_missing = any(
        phrase.lower() in answer.lower() for phrase in [
            "does not contain specific information",
            "doesn't contain specific information",
            "don't have specific information",
            "do not contain any information",
            "passages do not contain",
            "provided passages do not",
            "literature does not provide",
            "literature doesn't provide",
            "no specific information",
            "not found in the literature",
            "I apologize",
        ]
    )

    if sources and not acknowledged_missing:
        source_elements, source_actions, source_chunks = build_source_elements(reranked_points[:len(sources)])
        cl.user_session.set("source_chunks", source_chunks)
        response = answer
    else:
        source_elements = []
        source_actions = []
        if not sources:
            response = answer + "\n\n---\n\n**Note:** No relevant passages were found in the literature database."
        else:
            response = answer

    msg.content = response
    msg.elements = source_elements
    msg.actions = source_actions

    await msg.update()


# ============================================================================
# ACTION CALLBACKS
# ============================================================================


