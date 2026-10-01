from src.pipeline import build_ensemble_retriever
from src.retrieval.query_expansion import expand_query


def test_hybrid_search():
    print("=== INITIALIZING HYBRID SEARCH ===")
    ensemble_retriever = build_ensemble_retriever()

    original_query = "What are the limits and guidelines for KYC verification?"
    print(f"\nOriginal Query: '{original_query}'")

    print("\n1. Pushing query to Groq for legal translation...")
    optimized_query = expand_query(original_query)
    print(f"Optimized Search String: '{optimized_query}'")

    print("\n2. Executing Ensemble Search with optimized string...")
    results = ensemble_retriever.invoke(optimized_query)

    print("\n=== HYBRID RETRIEVAL RESULTS ===")
    for i, doc in enumerate(results[:3], 1):
        print(f"\n--- Top Match {i} ---")
        source = doc.metadata.get("source", "Unknown Document")
        headers = {k: v for k, v in doc.metadata.items() if k.startswith("Header")}
        print(f"Source Document: {source}")
        if headers:
            print(f"Regulatory Section: {headers}")

        print(f"\nContent Preview:\n{doc.page_content[:400]}...")


if __name__ == "__main__":
    test_hybrid_search()