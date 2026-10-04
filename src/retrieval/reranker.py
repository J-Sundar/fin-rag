import logging
from typing import List, Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.generation.llm_chain import _get_top_header
from src.utils.config import RERANKER_MODEL_NAME, TOP_K

logger = logging.getLogger(__name__)

_reranker_instance: Optional[CrossEncoder] = None


def get_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    """Returns a singleton CrossEncoder instance."""
    global _reranker_instance
    if _reranker_instance is None:
        logger.info(f"Loading CrossEncoder reranker model: {model_name}...")
        _reranker_instance = CrossEncoder(model_name)
    return _reranker_instance


def rerank_documents(
    query: str,
    docs: List[Document],
    top_k: int = TOP_K,
    model: Optional[CrossEncoder] = None,
) -> List[Document]:
    """
    Reranks documents against the query using a cross-encoder model.
    Enriches the document text with source and section metadata for context.
    """
    if not docs:
        return []

    if model is None:
        model = get_reranker()

    pairs = [
        [
            query,
            f"Document: {d.metadata.get('source', '')}\nSection: {_get_top_header(d.metadata)}\n{d.page_content}",
        ]
        for d in docs
    ]

    scores = model.predict(pairs)

    scored_docs = []
    for score, doc in zip(scores, docs):
        doc.metadata["rerank_score"] = float(score)
        scored_docs.append((float(score), doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

