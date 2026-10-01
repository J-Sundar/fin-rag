import logging
from typing import Optional, Tuple
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from src.ingestion.chunker import process_markdown_chunks
from src.retrieval.vector_store import get_vector_store
from src.retrieval.query_expansion import expand_query
from src.retrieval.reranker import rerank_documents
from src.generation.llm_chain import get_llm_chain, format_context
from src.utils.config import (
    PROCESSED_DIR,
    TOP_K,
    RETRIEVAL_CANDIDATES_K,
    ENSEMBLE_WEIGHTS,
)

logger = logging.getLogger(__name__)


def build_ensemble_retriever(
    k: int = RETRIEVAL_CANDIDATES_K,
    weights: Optional[list[float]] = None,
) -> EnsembleRetriever:
    """Initializes and returns hybrid BM25 + Qdrant ensemble retriever."""
    if weights is None:
        weights = ENSEMBLE_WEIGHTS

    logger.info("Loading chunks for BM25 retriever...")
    chunks = process_markdown_chunks(str(PROCESSED_DIR))
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    logger.info("Connecting to Qdrant vector store...")
    vector_store = get_vector_store()
    qdrant_retriever = vector_store.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, qdrant_retriever],
        weights=weights,
    )


def load_pipeline(
    k: int = RETRIEVAL_CANDIDATES_K,
    weights: Optional[list[float]] = None,
) -> Tuple[EnsembleRetriever, any]:
    """Loads and returns the ensemble retriever and generation chain."""
    ensemble_retriever = build_ensemble_retriever(k=k, weights=weights)
    llm_chain = get_llm_chain()
    return ensemble_retriever, llm_chain


def query_rag_pipeline(
    question: str,
    retriever: Optional[EnsembleRetriever] = None,
    llm_chain: Optional[any] = None,
    expand: bool = True,
    rerank: bool = True,
    k: int = TOP_K,
) -> dict:
    """Executes the full RAG pipeline: expansion -> retrieval -> reranking -> generation."""
    if retriever is None or llm_chain is None:
        retriever, llm_chain = load_pipeline(k=RETRIEVAL_CANDIDATES_K if rerank else k)

    search_query = expand_query(question) if expand else question

    if rerank:
        candidate_docs = retriever.invoke(search_query)[:RETRIEVAL_CANDIDATES_K]
        retrieved_docs = rerank_documents(query=question, docs=candidate_docs, top_k=k)
    else:
        candidate_docs = []
        retrieved_docs = retriever.invoke(search_query)[:k]

    context_text = format_context(retrieved_docs)
    answer = llm_chain.invoke({"context": context_text, "question": question})

    return {
        "question": question,
        "search_query": search_query,
        "candidate_docs": candidate_docs,
        "retrieved_docs": retrieved_docs,
        "context_text": context_text,
        "answer": answer,
    }

