import chainlit as cl
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import requests
from config import RETRIEVAL_CONFIG, LLM_CONFIG

# ============================================================================
# DEMO HELPER FUNCTIONS
# ============================================================================


def format_sources_with_scores(results):
    """
    Format sources with visual score bars
    
    Args:
        results: List of Qdrant search results
        
    Returns:
        str: Formatted sources text with scores
    """
    sources_text = "**Sources consulted:**\n"
    
    for i, result in enumerate(results, 1):
        
        sources_text += (
            f"[{i}] {os.path.basename(result.payload['source'])} "
            f"(Page {result.payload['page']}) - "
            f"Score: {result.score:.4f}\n"
        )
    
    return sources_text


# ============================================================================
# INITIALIZATION
# ============================================================================

load_dotenv()

# Initialize once at startup
print("Initializing XPCS Hypothesis Evaluator...")
embeddings = HuggingFaceEmbeddings(
    model_name="allenai/scibert_scivocab_uncased",
    model_kwargs={'device': 'cpu'}
)

client = QdrantClient(
    host=os.getenv('QDRANT_HOST', 'localhost'),
    port=int(os.getenv('QDRANT_PORT', 6333))
)

# Argo API configuration
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
        "max_tokens": LLM_CONFIG['max_tokens']
    }
    
    try:
        response = requests.post(ARGO_API_URL, json=payload, timeout=60)
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


# ============================================================================
# CHAINLIT HANDLERS
# ============================================================================

@cl.on_chat_start
async def start():
    # Initialize conversation history in user session
    cl.user_session.set("conversation_history", [])
    
    # System prompt - BALANCED VERSION
    system_prompt = {
        "role": "system",
        "content": """You are Argo, an expert AI assistant for X-ray Photon Correlation Spectroscopy (XPCS) at Argonne National Laboratory's Advanced Photon Source.

**Your Primary Responsibilities:**
- Answer questions about XPCS theory, techniques, and applications
- Help evaluate experimental hypotheses and feasibility
- Provide guidance on sample requirements and experimental design
- Maintain conversation context for follow-up questions

**CRITICAL RULES - Information Integrity:**

1. **Source Attribution:**
   - Build your answer PRIMARILY from the provided context passages
   - Quote or paraphrase specific sentences from the passages
   - Include formulas and definitions EXACTLY as written in the context
   - Cite sources using [Source N] format immediately after each claim
   - When a passage contains a mathematical formula, include it verbatim

2. **Using the Context Effectively:**
   - If passages discuss concepts related to the question, USE THEM
   - Synthesize information from multiple passages when relevant
   - Do NOT claim "the literature doesn't provide information" if the passages clearly address the topic
   - Example: If passages discuss speckle patterns, coherence, and dynamics → that IS information about XPCS

3. **When to Acknowledge Missing Information:**
   ONLY claim information is missing when:
   - The question asks for beamline-specific specifications (flux, energy range, detector model, sample environments)
   - The question asks about experimental protocols not described in the passages
   - The question asks about other facilities (LCLS, ESRF, etc.)
   - The retrieved passages have very low relevance scores and don't address the topic
   
   DO NOT claim information is missing when:
   - Passages describe the technique, even without a "textbook definition"
   - Passages contain formulas, experimental details, or theoretical concepts
   - Multiple passages discuss related aspects of the question

4. **Mathematical Formatting:**
   - Use LaTeX for all mathematical expressions
   - Inline math: $expression$
   - Display math (for important equations): $$expression$$
   - Examples:
     * "The speckle contrast is defined as $\\beta = \\sigma^2/\\langle I \\rangle^2$"
     * For key equations, use display mode:
       $$P(I) = \\frac{\\exp(-I/\\langle I \\rangle)}{\\langle I \\rangle}$$

5. **Citation Best Practices:**
   - Cite sources INLINE as you make claims, not just at the end
   - Be specific: "The scattering volume should be comparable to the coherence volume [Source 4]"
   - NOT vague: "XPCS involves speckle patterns [Source 1, 2, 3, 4, 5]"

**Example of CORRECT behavior:**

Question: "What is XPCS?"
Context: Contains passages about speckle patterns, coherence, thermodynamic fluctuations, experimental requirements
Response: "X-ray Photon Correlation Spectroscopy (XPCS) is a technique that directly measures thermodynamic fluctuations in material structure [Source 4]. When coherent X-rays scatter from a sample, they produce speckle patterns whose statistics reveal the system's dynamics [Source 3]..."

**Example of INCORRECT behavior:**

Question: "What is XPCS?"
Context: Contains passages about speckle patterns, coherence, dynamics
Response: "The retrieved literature does not provide a direct definition of XPCS. However, based on general principles..." ❌ WRONG! The passages DO provide information!

**Example of CORRECT acknowledgment of missing info:**

Question: "What is the photon flux at beamline 8-ID-I?"
Context: Contains general XPCS theory but no beamline specifications
Response: "The retrieved literature doesn't contain specific photon flux values for beamline 8-ID-I. For beamline specifications, please consult the beamline documentation or contact beamline staff directly."

**Tone:** Professional and helpful, suitable for users from students to senior scientists."""
    }
    
    cl.user_session.set("system_prompt", system_prompt)
    
    await cl.Message(
        content="👋 **Welcome to the XPCS Hypothesis Evaluator!**\n\n"
                "I can help you with questions about X-ray Photon Correlation Spectroscopy.\n\n"
                "**Ask me about:**\n"
                "- XPCS theory and principles\n"
                "- Experimental requirements\n"
                "- Sample preparation\n"
                "- Data analysis techniques\n\n"
                "📚 **Database:** 113 XPCS papers with 5,743 searchable passages\n\n"
                "💡 **Tip:** I'll show confidence levels and sources for transparency!\n\n"
                "I will remember our conversation, so feel free to ask follow-up questions!"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="🔍 Searching XPCS literature and generating answer...")
    await msg.send()
    
    # Get conversation history
    conversation_history = cl.user_session.get("conversation_history")
    system_prompt = cl.user_session.get("system_prompt")
    
    # Search for relevant context
    query_vector = embeddings.embed_query(message.content)
    results = client.query_points(
        collection_name=os.getenv('QDRANT_COLLECTION_NAME', 'xpcs_documents'),
        query=query_vector,
        limit=RETRIEVAL_CONFIG['num_results']
    )
    
    # Filter by relevance threshold and format context
    context_parts = []
    sources = []

    # DEBUG: Print all scores
    print(f"\n{'='*60}")
    print(f"Query: {message.content}")
    print(f"Retrieved {len(results.points)} results:")
    for idx, result in enumerate(results.points, 1):
        print(f"  [{idx}] Score: {result.score:.4f} | {os.path.basename(result.payload['source'])}")
    print(f"Threshold: {RETRIEVAL_CONFIG['relevance_threshold']}")
    print(f"{'='*60}\n")
    
    for idx, result in enumerate(results.points, 1):
        if result.score < RETRIEVAL_CONFIG['relevance_threshold']:
            continue
            
        source = os.path.basename(result.payload['source'])
        page = result.payload['page']
        text = result.payload['text']
        score = result.score
        
        context_parts.append(f"[Source {idx}: {source}, Page {page}]\n{text}")
        sources.append(f"[{idx}] {source} (Page {page})")
    
    # Store context and results for "Show Context" button
    cl.user_session.set("last_context", context_parts)
    cl.user_session.set("last_results", results.points)
    
    # Build context string
    if context_parts:
        context = "\n\n".join(context_parts)
        context_message = f"""You have been provided with relevant excerpts from XPCS scientific literature below.

CRITICAL INSTRUCTIONS:
1. Build your answer from these passages
2. Quote or paraphrase specific sentences
3. Include formulas exactly as written
4. Cite sources inline using [Source N]
5. Use LaTeX for mathematical expressions
6. IMPORTANT: When citing [Source N], make sure N matches the source number in the context below
7. DO NOT cite a source number that doesn't appear in the context

Context from XPCS literature:

{context}

User question: {message.content}

Provide a comprehensive answer based on the passages above. Make it clear which passage supports each claim. Double-check that your [Source N] citations match the source numbers in the context."""
        
    else:
        context_message = f"""No highly relevant passages found in the XPCS literature database (all results below {RETRIEVAL_CONFIG['relevance_threshold']:.0%} relevance threshold).

User question: {message.content}

Since no relevant passages were retrieved, clearly state:
"The literature database doesn't contain specific information about this topic. For information about [topic], please consult [appropriate resource]."

Do NOT attempt to answer from general knowledge."""
    
    # Build messages for Argo API
    messages = [system_prompt]
    
    # Add conversation history (last 5 exchanges to keep context manageable)
    messages.extend(conversation_history[-10:])  # Last 5 Q&A pairs
    
    # Add current question with context
    messages.append({
        "role": "user",
        "content": context_message
    })
    
    # Call Argo LLM
    answer = call_argo_llm(messages)
    
    # Update conversation history
    conversation_history.append({
        "role": "user",
        "content": message.content
    })
    conversation_history.append({
        "role": "assistant",
        "content": answer
    })
    cl.user_session.set("conversation_history", conversation_history)
    
    # Format sources with scores
    sources_with_scores = format_sources_with_scores(results.points[:len(sources)])
    
    # ========================================================================
    # Check if LLM acknowledged missing information
    # ========================================================================
    missing_info_phrases = [
        "does not contain specific information",
        "doesn't contain specific information",
        "literature does not provide",
        "literature doesn't provide",
        "no specific information",
        "not found in the literature"
    ]
    
    acknowledged_missing = any(phrase.lower() in answer.lower() for phrase in missing_info_phrases)
    
    # ========================================================================
    # Conditionally create "Show Context" button
    # ========================================================================
    if acknowledged_missing:
        # Don't show context button if LLM said info is missing
        actions = []
    else:
        # Show context button if LLM used the context
        actions = [
            cl.Action(
                name="show_context",
                payload={"action": "show_context"},
                label="📄 Show Retrieved Context",
                description="See the exact passages used to generate this answer"
            )
        ]
    
    # ========================================================================
    # Format final response WITHOUT confidence indicator
    # ========================================================================
    if sources:
        response = f"""{answer}

---

{sources_with_scores}"""
    else:
        response = f"""{answer}

---

**Note:** No passages met the relevance threshold of {RETRIEVAL_CONFIG['relevance_threshold']:.0%}. Answer based on general XPCS knowledge."""
    
    msg.content = response
    msg.actions = actions
    
    await msg.update()


# ============================================================================
# ACTION CALLBACKS (MUST BE AT MODULE LEVEL, NOT INSIDE @cl.on_message)
# ============================================================================

@cl.action_callback("show_context")
async def on_show_context(action):
    """
    Handle the "Show Retrieved Context" button click
    """
    context_parts = cl.user_session.get("last_context")
    results = cl.user_session.get("last_results")
    
    if not context_parts:
        await cl.Message(
            content="⚠️ No context available. Please ask a question first."
        ).send()
        return
    
    # Format the context nicely
    context_display = "# 📄 Retrieved Context Passages\n\n"
    context_display += "="*80 + "\n\n"
    
    for i, (part, result) in enumerate(zip(context_parts, results), 1):
        context_display += f"## Passage {i} (Score: {result.score:.4f})\n\n"
        context_display += f"{part}\n\n"
        context_display += "-"*80 + "\n\n"
    
    await cl.Message(
        content=context_display
    ).send()
