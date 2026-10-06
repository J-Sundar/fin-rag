"""
src/retrieval/vector_store.py

Manages the connection to the Qdrant Vector Database, allowing us to
ingest embedded documents and perform semantic searches.
"""

import logging
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from src.embedding.embedder import get_embedding_model
from src.utils.config import (
    QDRANT_DB_PATH,
    PROCESSED_DIR,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    TOP_K,
)

logger = logging.getLogger(__name__)


def build_vector_store(documents) -> QdrantVectorStore:
    """
    Takes a list of chunked LangChain Documents, embeds them, and
    saves them to a persistent local Qdrant database.
    """
    embeddings = get_embedding_model()

    logger.info(f"Connecting to Qdrant at {QDRANT_DB_PATH}")
    client = QdrantClient(path=str(QDRANT_DB_PATH))

    if client.collection_exists(COLLECTION_NAME):
        logger.info(f"Collection '{COLLECTION_NAME}' exists. Recreating for a fresh build...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    logger.info(f"Embedding {len(documents)} chunks and pushing to Qdrant...")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_documents(documents)

    logger.info("Vector database build complete.")
    return vector_store


def get_vector_store() -> QdrantVectorStore:
    """
    Returns a connection to the existing Qdrant collection for querying.
    If the collection does not exist, automatically builds it from processed markdown chunks.
    """
    client = QdrantClient(path=str(QDRANT_DB_PATH))

    if not client.collection_exists(COLLECTION_NAME):
        logger.info(f"Collection '{COLLECTION_NAME}' not found. Initializing from {PROCESSED_DIR}...")
        from src.ingestion.chunker import process_markdown_chunks
        chunks = process_markdown_chunks(str(PROCESSED_DIR))
        return build_vector_store(chunks)

    embeddings = get_embedding_model()
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )