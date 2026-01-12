# Retrieval Configuration
RETRIEVAL_CONFIG = {
    # Number of passages to retrieve from vector database
    'num_results': 7,
    
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
    # Argo model to use
    'model': 'gpt4o',
    
    # Temperature (0-2): Higher = more creative, Lower = more focused
    'temperature': 0.7,
    
    # Max tokens in response
    'max_tokens': 2000,
    
    # Number of conversation turns to remember (each turn = 1 Q&A pair)
    'conversation_memory': 5,
}