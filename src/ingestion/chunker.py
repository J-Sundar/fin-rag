"""
src/ingestion/chunker.py

Reads processed Markdown files and splits them into semantic chunks 
based on document headers, preserving the structural context of regulatory 
documents. Each chunk is also tagged with an approximate source page number,
derived from the page-break markers Docling embeds during PDF conversion
(see pdf_loader.py) — this is what lets the UI show "Source, p.12" style
citations instead of just a filename.
"""

import logging
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.utils.config import PAGE_BREAK_MARKER, CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _attach_page_numbers(header_doc: Document, sub_docs: list[Document], doc_start_page: int) -> None:
    """
    Mutates sub_docs in place: tags each with metadata["page"] (the page it
    most likely starts on) and strips the page-break marker back out of the
    visible chunk text.

    Page numbers are approximate, not pixel-exact: for a chunk that straddles
    a page boundary, we assign the page it *starts* on. Good enough for a
    citation like "p.12" — not meant for exact-offset legal reference.

    Works by re-locating each sub-chunk's text inside the parent header_doc's
    content (searching forward from a rolling cursor, offset back by the
    overlap window so repeated/overlapping text is still found) and counting
    how many page-break markers precede that position.
    """
    content = header_doc.page_content
    cursor = 0
    for sub in sub_docs:
        search_from = max(0, cursor - CHUNK_OVERLAP - 20)
        idx = content.find(sub.page_content, search_from)
        if idx == -1:
            # Splitter trimmed whitespace or otherwise altered the text enough
            # that we couldn't relocate it exactly — fall back to "wherever
            # we left off" rather than guessing page 1.
            idx = cursor
        markers_before = content[:idx].count(PAGE_BREAK_MARKER)
        sub.metadata["page"] = doc_start_page + markers_before
        sub.page_content = sub.page_content.replace(PAGE_BREAK_MARKER, " ").strip()
        cursor = idx + len(sub.page_content)


def process_markdown_chunks(processed_dir: str) -> list[Document]:
    """
    Reads all Markdown files in a directory and splits them into logical 
    chunks based on headers, falling back to character splitting for massive 
    sections. Each chunk carries source filename + approximate page metadata.
    """
    md_dir = Path(processed_dir)
    md_files = sorted(md_dir.glob("*.md"))

    if not md_files:
        logger.warning(f"No Markdown files found in: {md_dir}")
        return []

    # 1. Define the header hierarchy we want to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # 2. Define a fallback splitter for extremely long sections under a single header
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []

    for md_path in md_files:
        logger.info(f"Chunking: {md_path.name}")
        
        with open(md_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        if PAGE_BREAK_MARKER not in markdown_text:
            logger.warning(
                f"  No page-break markers found in {md_path.name} — it was likely "
                f"generated before the page-citation fix. Delete it and re-run "
                f"build_db.py to regenerate with page numbers, otherwise every "
                f"chunk from this file will default to page 1."
            )

        # Step A: Split by Markdown headers (page-break markers ride along
        # inside each split's page_content at this point)
        header_splits = markdown_splitter.split_text(markdown_text)

        # Step B: Walk header splits in document order. Each one knows its
        # starting page from how many markers preceded it; each one's own
        # marker count feeds forward into the next split's starting page.
        pages_consumed = 0
        doc_final_splits = []
        for header_doc in header_splits:
            doc_start_page = pages_consumed + 1
            sub_docs = text_splitter.split_documents([header_doc])
            _attach_page_numbers(header_doc, sub_docs, doc_start_page)
            doc_final_splits.extend(sub_docs)
            pages_consumed += header_doc.page_content.count(PAGE_BREAK_MARKER)

        # Step C: Drop any empty chunks (can happen when the fallback
        # splitter's overlap window lands entirely on a page-break marker)
        # and inject the source filename into the metadata for tracking.
        for chunk in doc_final_splits:
            if not chunk.page_content.strip():
                continue
            chunk.metadata["source"] = md_path.name
            all_chunks.append(chunk)

        logger.info(f"  → Generated {len(doc_final_splits)} chunks.")

    logger.info(f"Total chunks created across all documents: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    # Local testing execution
    PROCESSED_DATA_PATH = "../../data/processed"
    chunks = process_markdown_chunks(PROCESSED_DATA_PATH)
    
    # Print a sample to verify metadata routing is working
    if chunks:
        print("\n--- Sample Chunk ---")
        print(f"Metadata: {chunks[0].metadata}")
        print(f"Content Preview: {chunks[0].page_content[:200]}...")

        mid_idx = len(chunks) // 2
        print(f"\n--- Sample Chunk [{mid_idx}] ---")
        print(f"Metadata: {chunks[mid_idx].metadata}")
        print(f"Content Preview: {chunks[mid_idx].page_content[:200]}...")