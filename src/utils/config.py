"""
src/utils/config.py

Single source of truth for all project configuration.
Every module imports constants and env vars from here instead of
defining them inline. This means you change a value in one place
and it propagates everywhere automatically.
"""

import os
import logging
import torch
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
# load_dotenv() reads the .env file and injects its key=value pairs into
# os.environ. Calling it here once means no other module needs to call it.
# If .env doesn't exist (e.g. on Hugging Face Spaces where secrets are set
# as environment variables directly), load_dotenv() is a no-op — safe either way.
load_dotenv()

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
# Path(__file__) is the absolute path to THIS file (config.py).
# .parent gives src/utils/, .parent again gives src/, .parent again gives
# the project root. All other paths are derived from that anchor so the
# project works regardless of which directory you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RAW_PDF_DIR    = PROJECT_ROOT / "data" / "raw_pdfs"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
QDRANT_DB_PATH = PROJECT_ROOT / "data" / "qdrant_db"

# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------
COLLECTION_NAME    = "rbi_sebi_circulars"
EMBEDDING_DIM      = 384   # bge-small-en-v1.5 → 384  |  bge-base-en-v1.5 → 768
                           # ⚠ If you switch models, update this too.

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME   = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZE_EMBEDDINGS   = True     # Required for cosine similarity with Qdrant

# ---------------------------------------------------------------------------
# LLM (Groq)
# ---------------------------------------------------------------------------
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")   # Read from .env or host env vars
GROQ_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0   # Low temp = deterministic, factual answers

# The exact refusal string the LLM must emit when context is insufficient.
# Shared between llm_chain.py (the prompt instruction) and evaluate.py (the
# groundedness check) so the two can never drift out of sync.
REFUSAL_MESSAGE = "I cannot answer this based on the retrieved documents."

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 1024
CHUNK_OVERLAP = 100

# Docling embeds this literal string into the exported Markdown at every
# page boundary (see pdf_loader.py). The chunker counts occurrences of this
# marker to derive an approximate page number for each chunk, then strips
# it back out of the visible text.
PAGE_BREAK_MARKER = "<!-- DOCLING_PAGE_BREAK -->"

# ---------------------------------------------------------------------------
# Retrieval & Reranking
# ---------------------------------------------------------------------------
TOP_K = 5   # Number of chunks to pass to LLM context
RETRIEVAL_CANDIDATES_K = 15  # Candidate pool size retrieved prior to reranking
ENSEMBLE_WEIGHTS = [0.4, 0.6]  # [BM25 lexical weight, Qdrant semantic weight]
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Call setup_logging() at the top of any entry-point script (build_db.py,
# app.py, test scripts). Modules themselves only need logging.getLogger(__name__).
def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    )