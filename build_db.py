# build_db.py

import logging
from src.ingestion.pdf_loader import convert_pdfs_to_markdown
from src.ingestion.chunker import process_markdown_chunks
from src.retrieval.vector_store import build_vector_store
from src.utils.config import RAW_PDF_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_database_build():
    logger.info("=== Starting Database Build Pipeline ===")

    # 0. Convert any raw PDFs that don't already have a processed .md.
    #    convert_pdfs_to_markdown() skips files that already exist, so this
    #    is safe to call on every run — first run does real work, later runs
    #    are a no-op unless you add new PDFs or delete a processed .md.
    logger.info("Checking data/raw_pdfs for PDFs to convert...")
    convert_pdfs_to_markdown(str(RAW_PDF_DIR), str(PROCESSED_DIR))

    # 1. Grab our semantically chunked documents
    chunks = process_markdown_chunks(str(PROCESSED_DIR))

    if not chunks:
        logger.error(
            "No chunks found! Check that PDFs exist in data/raw_pdfs — "
            "nothing was there to convert."
        )
        return

    # 2. Embed and store them in Qdrant
    logger.info(f"Initializing embedding model and pushing {len(chunks)} chunks to Qdrant...")
    build_vector_store(chunks)

    logger.info("=== Database Build Complete! ===")

if __name__ == "__main__":
    run_database_build()