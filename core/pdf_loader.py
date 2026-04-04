import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import Set


def process_all_pdfs(pdf_directory: str, existing_files: Set[str] = None):
    """
    Process PDF files in a directory, skipping already-indexed ones.

    Args:
        pdf_directory: Path to the folder containing PDFs.
        existing_files: Set of filenames already present in the vector store.
                        Pass None or empty set to process everything.

    Returns:
        List of loaded LangChain Document objects (only from new PDFs).
    """
    if existing_files is None:
        existing_files = set()

    all_documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF file(s) in '{pdf_directory}'")

    skipped = 0
    for pdf_file in pdf_files:
        if pdf_file.name in existing_files:
            print(f"  ⏭  Skipping (already indexed): {pdf_file.name}")
            skipped += 1
            continue

        print(f"\n  Processing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} page(s)")

        except Exception as e:
            print(f"  ✗ Error loading {pdf_file.name}: {e}")

    print(f"\nSummary: {skipped} skipped | {len(pdf_files) - skipped} new | "
          f"{len(all_documents)} total pages loaded")
    return all_documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into smaller chunks for better RAG performance.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    if not documents:
        print("No new documents to split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} page(s) into {len(split_docs)} chunk(s)")

    if split_docs:
        print(f"\nExample chunk:")
        print(f"  Content : {split_docs[0].page_content[:200]}...")
        print(f"  Metadata: {split_docs[0].metadata}")

    return split_docs