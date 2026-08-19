"""
src/ingestion/pdf_loader.py

Parses complex regulatory PDFs (RBI/SEBI) using Docling to preserve 
tables, multi-column layouts, and hierarchies. Outputs clean Markdown.

A page-break marker is embedded at every page boundary during export so
that downstream chunking (chunker.py) can attach an approximate page
number to each chunk — this is what powers page-level citations in the
UI, since Markdown itself has no native concept of a "page".
"""

import logging
from pathlib import Path
from docling.document_converter import (
    DocumentConverter,
    InputFormat,
    PdfFormatOption,
)
from docling.datamodel.pipeline_options import PdfPipelineOptions
from src.utils.config import PAGE_BREAK_MARKER

# Set up logging for production-style tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_pdfs_to_markdown(raw_pdf_dir: str, processed_dir: str) -> None:
    """
    Scans a directory for raw PDFs, converts them to layout-aware Markdown 
    using Docling, and saves the output to a processed directory.

    Args:
        raw_pdf_dir: Path to the folder containing raw PDFs.
        processed_dir: Path to the folder to save Markdown files.
    """
    pdf_dir = Path(raw_pdf_dir)
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in: {pdf_dir}")
        return

    # Initialize the Docling converter
    logger.info("Initializing Docling DocumentConverter...")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    for pdf_path in pdf_files:
        md_filename = pdf_path.with_suffix('.md').name
        md_filepath = out_dir / md_filename

        # Skip if we already processed this file (saves time during testing)
        if md_filepath.exists():
            logger.info(f"Skipping {pdf_path.name} - Markdown already exists.")
            continue

        logger.info(f"Processing: {pdf_path.name}")
        
        try:
            # The core conversion step
            result = converter.convert(pdf_path)
            
            # Export to table-preserved Markdown, embedding a page-break
            # marker at every page boundary. Requires docling-core >= 2.24.0.
            # Note: page_break_placeholder is a static string, not a
            # per-page template — page *numbers* are derived later by the
            # chunker counting how many markers precede a given chunk.
            markdown_content = result.document.export_to_markdown(
                page_break_placeholder=PAGE_BREAK_MARKER
            )

            # Save the clean data to our processed folder
            with open(md_filepath, "w", encoding="utf-8") as f:
                f.write(markdown_content)
                
            logger.info(f"Successfully saved Markdown to {md_filename}")

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {str(e)}")

    logger.info("Ingestion complete.")

if __name__ == "__main__":
    # Local testing execution
    RAW_DATA_PATH = "../../data/raw_pdfs"
    PROCESSED_DATA_PATH = "../../data/processed"
    
    convert_pdfs_to_markdown(RAW_DATA_PATH, PROCESSED_DATA_PATH)