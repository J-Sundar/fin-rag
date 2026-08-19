# tests/test_retrieval.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from src.retrieval.vector_store import get_vector_store
from src.ingestion.chunker import process_markdown_chunks

load_dotenv()

def expand_query(original_query: str) -> str:
    """
    Uses Groq to translate a natural language query into an optimized 
    bag of legal keywords for the BM25 search engine.
    """
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    prompt = PromptTemplate.from_template("""
    You are an expert in Indian financial regulations (RBI/SEBI). 
    A user is asking a question. Your job is to rewrite this question into a list of 
    highly specific keywords, synonyms, and formal legalese that would appear in an official document.
    
    Rules:
    1. Convert colloquial terms to formal terms (e.g., "KYC" -> "Customer Due Diligence", "CDD").
    2. Convert numerical concepts to words (e.g., "limits" -> "exceeding", "rupees fifty thousand", "maximum").
    3. Output ONLY a space-separated string of keywords. Do NOT output full sentences.
    
    User Question: {question}
    
    Optimized Search Keywords:
    """)
    
    chain = prompt | llm | StrOutputParser()
    expanded_query = chain.invoke({"question": original_query})
    
    return expanded_query

def test_hybrid_search():
    print("=== INITIALIZING HYBRID SEARCH ===")
    
    vector_store = get_vector_store()
    qdrant_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    chunks = process_markdown_chunks("data/processed")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, qdrant_retriever],
        weights=[0.5, 0.5]
    )
    
    original_query = "What are the limits and guidelines for KYC verification?"
    print(f"\nOriginal Query: '{original_query}'")
    
    # --- THE MAGIC HAPPENS HERE ---
    print("\n1. Pushing query to Groq for legal translation...")
    optimized_query = expand_query(original_query)
    print(f"Optimized Search String: '{optimized_query}'")
    
    print("\n2. Executing Ensemble Search with optimized string...")
    results = ensemble_retriever.invoke(optimized_query)
    
    print("\n=== HYBRID RETRIEVAL RESULTS ===")
    for i, doc in enumerate(results[:3], 1):
        print(f"\n--- Top Match {i} ---")
        source = doc.metadata.get('source', 'Unknown Document')
        headers = {k: v for k, v in doc.metadata.items() if k.startswith('Header')}
        print(f"Source Document: {source}")
        if headers:
            print(f"Regulatory Section: {headers}")
            
        print(f"\nContent Preview:\n{doc.page_content[:400]}...")

if __name__ == "__main__":
    test_hybrid_search()