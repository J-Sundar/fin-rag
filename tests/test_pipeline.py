# test_pipeline.py

from src.retrieval.vector_store import get_vector_store
from src.generation.llm_chain import get_llm_chain

def run_full_pipeline(query: str):
    print(f"=== PROCESSING QUERY: '{query}' ===\n")
    
    # 1. Retrieval
    print("1. Querying Qdrant Vector Database...")
    vector_store = get_vector_store()
    retrieved_docs = vector_store.similarity_search(query, k=3)
    
    # 2. Context Formatting
    # Stitch the chunks together, prepending the source document name for the LLM
    context_chunks = []
    for doc in retrieved_docs:
        source = doc.metadata.get('source', 'Unknown Document')
        context_chunks.append(f"[Source: {source}]\n{doc.page_content}")
        
    context_text = "\n\n---\n\n".join(context_chunks)
    
    # 3. Generation
    print("2. Pushing context to Groq LLM...")
    chain = get_llm_chain()
    
    response = chain.invoke({
        "context": context_text, 
        "question": query
    })
    
    print("\n=== FINAL RAG OUTPUT ===")
    print(response)

if __name__ == "__main__":
    # Testing the same query we know has bad retrieval right now
    run_full_pipeline("What are the limits and guidelines for KYC verification?")