# test_ingestion.py

import os
from src.ingestion.pdf_loader import convert_pdfs_to_markdown
from src.ingestion.chunker import process_markdown_chunks
from src.utils.config import RAW_PDF_DIR, PROCESSED_DIR


def test_ingestion_pipeline():
    print("=== Phase 1: Testing PDF to Markdown Extraction ===")
    convert_pdfs_to_markdown(str(RAW_PDF_DIR), str(PROCESSED_DIR))
    
    md_files = os.listdir(PROCESSED_DIR)
    print(f"Files in processed directory: {md_files}\n")

    print("=== Phase 2: Testing Semantic Chunking ===")
    # This should read the .md file and split it by headers
    chunks = process_markdown_chunks(PROCESSED_DIR)
    
    print("\n=== Phase 3: Verification ===")
    print(f"Total chunks successfully generated: {len(chunks)}")
    
    if chunks:
        print("\n--- Inspecting Chunk ---")
        print(f"Metadata attached: {chunks[0].metadata}")
        print(f"Content preview:\n{chunks[0].page_content[:300]}...")
        
        # Let's also check a random chunk from the middle to ensure tables/headers held up
        mid_idx = len(chunks) // 2
        print(f"--- Inspecting Chunk [{mid_idx}] ---")
        print(f"Metadata attached: {chunks[mid_idx].metadata}")
        print(f"Content preview:\n{chunks[mid_idx].page_content[:300]}...\n")
