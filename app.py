from admin.admin import launch_admin
from admin.weights_manager import load_weights, save_weights, get_all_docs, apply_weights
launch_admin()

import os
import json
import requests

import chainlit as cl
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from config import RETRIEVAL_CONFIG, LLM_CONFIG

import ldap

import secrets
from auth_tokens import admin_auth_tokens, add_token, remove_token


# ============================================================================
# DEMO HELPER FUNCTIONS
# ============================================================================

import re


def clean_title(title):
    """Strip MathML/XML tags and HTML from CrossRef titles."""
    if not title:
        return title
    # remove XML/HTML tags
    title = re.sub(r'<[^>]+>', '', title)
    # collapse extra whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def format_sources_with_scores(results):
    """Format sources with full citation metadata"""
    sources_text = "**Sources consulted:**\n\n"

    seen = set()
    num = 0

    for result in results:
        p = result.payload

        title   = clean_title(p.get("title")) or os.path.basename(p["source"])
        authors = p.get("authors") or []
        journal = p.get("journal") or ""
        year    = p.get("year")    or ""
        url     = p.get("url")     or ""
        doi     = p.get("doi")     or ""
        page    = p.get("page", "")

        if title in seen:
            continue
        seen.add(title)
        num += 1

        # author string
        if len(authors) > 2:
            author_str = f"{authors[0]} et al."
        elif authors:
            author_str = ", ".join(authors)
        else:
            author_str = ""

        # build entry
        sources_text += "---\n"
        sources_text += f"**{title}**  \n"

        if author_str:
            sources_text += f"**Author(s):** {author_str}  \n"
        if journal:
            sources_text += f"**Journal:** {journal}  \n"
        if year:
            sources_text += f"**Year:** {year}  \n"
        if page:
            sources_text += f"**Page:** {page}  \n"
        if doi and url:
            sources_text += f"**DOI:** [{doi}]({url})  \n"

        sources_text += "\n"

    return sources_text


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
        response = requests.post(ARGO_API_URL, json=payload, timeout=90)
        if not response.ok:
            print(f"Argo API Error {response.status_code}: {response.text}")
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
        return f"Network error calling Argo API: {str(e)}"
    except Exception as e:
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


def rerank_chunks(question: str, candidates: list) -> list:
    """Use LLM to filter retrieved chunks down to those that actually answer the question."""
    if not candidates:
        return []

    passages = []
    for i, point in enumerate(candidates, 1):
        text = point.payload.get("text", "")[:600]
        passages.append(str(i) + ". " + text)

    passages_str = "\n\n".join(passages)

    prompt = (
        "You are a relevance filter for a scientific Q&A system about XPCS "
        "(X-ray Photon Correlation Spectroscopy).\n\n"
        "Question: " + question + "\n\n"
        "Below are " + str(len(candidates)) + " passages retrieved from scientific literature. "
        "Identify which passages contain information that directly helps answer the question. "
        "Be strict — only select passages that contain relevant information.\n\n"
        + passages_str + "\n\n"
        "Respond with ONLY valid JSON listing the relevant passage numbers.\n"
        "Example: {\"relevant\": [1, 3, 5]}\n"
        "If none are relevant: {\"relevant\": []}"
    )

    payload = {
        "user": ARGO_USER,
        "model": LLM_CONFIG["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": 150,
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

        match = re.search(r'\{[^}]*"relevant"[^}]*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            indices = data.get("relevant", [])
            if isinstance(indices, list) and indices:
                filtered = [candidates[i - 1] for i in indices if 1 <= i <= len(candidates)]
                print("[RERANKER] Kept", len(filtered), "of", len(candidates), "chunks")
                return filtered

        print("[RERANKER] Could not parse response, using all candidates:", text[:100])
    except Exception as e:
        print("[RERANKER] Error:", e, "— falling back to all candidates")

    return candidates


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
            return None

        try:
            conn.simple_bind_s(user_dn, password)
        except ldap.INVALID_CREDENTIALS:
            print(f"[AUTH] Invalid password for: {username}")
            return None

        first_name = user_info.get("givenName", [b""])[0].decode()
        last_name = user_info.get("sn", [b""])[0].decode()
        email = user_info.get("mail", [b""])[0].decode()

        print(f"[AUTH] Login successful: {username} ({first_name} {last_name})")

        return cl.User(
            identifier=username,
            metadata={"name": f"{first_name} {last_name}", "email": email}
        )

    except Exception as e:
        print(f"[AUTH ERROR] {e}")
        return None


@cl.on_chat_start
async def start():

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
   - Cite sources inline by title: "XPCS measures dynamics via speckle fluctuations [Source: X-ray photon correlation spectroscopy]"
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
    token = cl.user_session.get("admin_token")
    if token:
        remove_token(token)
        print(f"[AUTH] Token invalidated on session end")

@cl.on_message
async def main(message: cl.Message):

    msg = cl.Message(content="Searching XPCS literature...")
    await msg.send()
    
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

    # Tertiary: adjacent chunk retrieval — for each retrieved doc, also fetch neighboring pages
    # so that definitions/context on adjacent pages aren't missed
    doc_pages = {}
    for point in combined_points:
        src = point.payload.get('source', '')
        page = point.payload.get('page')
        if src and page is not None:
            doc_pages.setdefault(src, set()).add(int(page))

    adj_added = 0
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
                combined_points.append(_SyntheticPoint(record, score=0.80))
                seen_ids.add(record.id)
                adj_added += 1

    print("Keywords:", keywords, "| kw added:", kw_added, "| adjacent added:", adj_added)

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
    msg.content = "Verifying source relevance..."
    await msg.update()

    reranked_points = rerank_chunks(message.content, reranked_points)

    context_parts = []
    sources = []
    seen_titles = set()

    for idx, result in enumerate(reranked_points, 1):
        p = result.payload
        source = os.path.basename(p['source'])
        page = p['page']
        text = p['text']
        title = p.get('title') or source
        weight = current_weights.get(source, 50)

        context_parts.append(
            f"[{title}, Page {page}, Priority: {weight}/100]\n{text}"
        )
        sources.append(f"[{idx}] {title} (Page {page})")
        seen_titles.add(title)

    cl.user_session.set("last_context", context_parts)
    cl.user_session.set("last_results", reranked_points[:len(sources)])

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
5. Cite sources inline using the document title in brackets, e.g. [Source: X-ray photon correlation spectroscopy]
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
    
    # Call Argo LLM
    answer = call_argo_llm(messages)
    
    # Update conversation history
    conversation_history.append({"role": "user", "content": message.content})
    conversation_history.append({"role": "assistant", "content": answer})
    cl.user_session.set("conversation_history", conversation_history)
    
    # Format sources
    sources_with_scores = format_sources_with_scores(reranked_points[:len(sources)])
    
    # Check if LLM acknowledged missing information or out of scope question
    missing_info_phrases = [
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
        "I apologize"
    ]
    
    acknowledged_missing = any(phrase.lower() in answer.lower() for phrase in missing_info_phrases)
    
    # Conditionally create "Show Context" button
    if acknowledged_missing or not sources:
        actions = []
        sources_with_scores = ""
    else:
        actions = [
            cl.Action(
                name="show_context",
                payload={"action": "show_context"},
                label="📄 Show Retrieved Context",
                description="See the exact passages used to generate this answer"
            )
        ]
    
    # Format final response
    if sources:
        response = f"""{answer}

---

{sources_with_scores}"""
    else:
        response = answer + "\n\n---\n\n**Note:** No relevant passages were found in the literature database."
    
    msg.content = response
    msg.actions = actions
    
    await msg.update()


# ============================================================================
# ACTION CALLBACKS
# ============================================================================

@cl.action_callback("show_context")
async def on_show_context(action):
    """Handle the "Show Retrieved Context" button click"""
    context_parts = cl.user_session.get("last_context")
    results = cl.user_session.get("last_results")

    if not context_parts:
        await cl.Message(
            content="No context available. Please ask a question first."
        ).send()
        return

    context_display = "# Retrieved Context Passages\n\n"
    context_display += "*These are the raw text chunks retrieved from the vector database that were used to generate the answer above.*\n\n"
    context_display += "=" * 65 + "\n\n"

    for part, result in zip(context_parts, results):
        context_display += f"{part}\n\n"
        context_display += "-" * 80 + "\n\n"

    await cl.Message(content=context_display).send()

