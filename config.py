# Retrieval Configuration
RETRIEVAL_CONFIG = {
    # Number of passages to retrieve from vector database
    'num_results': 40,
    
    # Minimum similarity score (0-1) to include a passage
    # Higher = more strict, only very relevant passages
    # Lower = more permissive, includes somewhat relevant passages
    'relevance_threshold': 0.01,
    
    # Chunk size for document splitting (already set during ingestion)
    # To change this, you need to re-run ingest_documents.py
    'chunk_size': 1000,
    'chunk_overlap': 200,
    
}

# LLM Configuration
LLM_CONFIG = {
    'model': 'claudesonnet4',
    'temperature': 0.7,
    'top_p': 0.9,
    'max_tokens': 2000,        # Anthropic models use max_tokens
    'conversation_memory': 5,
}

# Reranker uses a faster/cheaper model — only needs to output a JSON index list
RERANKER_CONFIG = {
    'model': 'gpt41nano',      # fastest GPT-4.1 model; swap for any model available on Argo
    'max_candidates': 100,     # top-N by score sent to reranker
    'preview_chars': 800,      # chars per chunk shown to reranker (chunk size is 1000)
    'max_tokens': 800,
}
