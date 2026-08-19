"""
src/embedding/embedder.py

Handles loading the open-source BAAI embedding model and
converting LangChain documents into dense vectors.
"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
)

logger = logging.getLogger(__name__)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the configured BAAI embedding model.
    Runs entirely locally — no API calls.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} on {EMBEDDING_DEVICE}")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS},
    )