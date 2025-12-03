import chainlit as cl
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import requests
from config import RETRIEVAL_CONFIG, LLM_CONFIG

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
ARGO_API_URL = "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"
ARGO_USER = os.getenv('ARGO_USER', 'your_anl_username')

print("Ready!")

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

@cl.on_chat_start
async def start():
    # Initialize conversation history in user session
    cl.user_session.set("conversation_history", [])
    
    # System prompt
    system_prompt = {
        "role": "system",
        "content": """You are an expert assistant for X-ray Photon Correlation Spectroscopy (XPCS) 
at Argonne National Laboratory's Advanced Photon Source, beamline 8-ID.

Your role is to:
1. Answer questions about XPCS theory, techniques, and applications
2. Help users evaluate experimental hypotheses and feasibility
3. Provide guidance on sample requirements and experimental design
4. Base your answers on the provided scientific literature context
5. Cite sources when possible using the [Source N] references
6. Be honest when information is not available in the context
7. Remember previous questions in the conversation to provide contextual answers

Maintain a professional, helpful tone suitable for beamline users ranging from students to senior scientists."""
    }
    
    cl.user_session.set("system_prompt", system_prompt)
    
    await cl.Message(
        content="Welcome to the XPCS Hypothesis Evaluator!\n\n"
                "I can help you with questions about X-ray Photon Correlation Spectroscopy.\n\n"
                "Ask me anything about:\n"
                "- XPCS theory and principles\n"
                "- Experimental requirements\n"
                "- Feasibility at beamline 8-ID\n"
                "- Sample preparation\n"
                "- Data analysis techniques\n\n"
                "Database contains 113 XPCS papers with 5743 searchable passages.\n\n"
                "I will remember our conversation, so feel free to ask follow-up questions!"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="Searching XPCS literature and generating answer...")
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
    
    for idx, result in enumerate(results.points, 1):
        if result.score < RETRIEVAL_CONFIG['relevance_threshold']:
            continue
            
        source = os.path.basename(result.payload['source'])
        page = result.payload['page']
        text = result.payload['text']
        score = result.score
        
        context_parts.append(f"[Source {idx}: {source}, Page {page}]\n{text}")  # Removed score from context
        sources.append(f"[{idx}] {source} (Page {page})")  # Removed score from display
    
    # Build context string
    if context_parts:
        context = "\n\n".join(context_parts)
        context_message = f"""You have been provided with relevant excerpts from XPCS scientific literature below. Use this information to answer the user's question.

    Context from XPCS literature:

    {context}

    User question: {message.content}

    Instructions: Provide a comprehensive answer based on the context above. Cite sources using [Source N] notation. The context IS relevant to XPCS - use it to inform your answer."""
    else:
        context_message = f"""No highly relevant passages found in the XPCS literature database (all results below {RETRIEVAL_CONFIG['relevance_threshold']:.0%} relevance threshold).

    User question: {message.content}

    Please provide a general answer based on your knowledge of XPCS, but clearly state that this is not based on the specific literature in the database."""
    
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
    
    # Format final response with sources
    if sources:
        response = f"{answer}\n\n---\n\n**Sources consulted:**\n"
        for source in sources:
            response += f"- {source}\n"
    else:
        response = f"{answer}\n\n---\n\n**Note:** No passages met the relevance threshold of {RETRIEVAL_CONFIG['relevance_threshold']:.0%}. Answer based on general XPCS knowledge."
    
    msg.content = response
    await msg.update()