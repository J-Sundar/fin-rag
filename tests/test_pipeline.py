from src.pipeline import query_rag_pipeline


def run_full_pipeline(query: str):
    print(f"=== PROCESSING QUERY: '{query}' ===\n")
    result = query_rag_pipeline(query)

    print(f"Expanded Query: {result['search_query']}\n")
    print(f"Retrieved {len(result['retrieved_docs'])} docs.\n")
    print("=== FINAL RAG OUTPUT ===")
    print(result["answer"])


if __name__ == "__main__":
    run_full_pipeline("What are the limits and guidelines for KYC verification?")