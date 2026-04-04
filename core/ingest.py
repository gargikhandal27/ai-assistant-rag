"""
ingest.py — Run this script to load new PDFs into the vector store.
Already-indexed files are automatically skipped.
"""

from pdf_loader import process_all_pdfs, split_documents
from embedding_manager import EmbeddingManager
from vector_store import VectorStore

PDF_DIRECTORY = "data/pdf"


def ingest_pdfs():
    # 1. Connect to the (persistent) vector store
    vector_store = VectorStore(
        collection_name="pdf_documents",
        persist_directory="data/vector_store"
    )

    # 2. Find out which files are already indexed
    existing_files = vector_store.get_existing_files()
    if existing_files:
        print(f"\nAlready indexed: {existing_files}\n")
    else:
        print("\nNo files indexed yet — fresh start.\n")

    # 3. Load only NEW PDFs (skips already-indexed ones)
    new_documents = process_all_pdfs(PDF_DIRECTORY, existing_files=existing_files)

    if not new_documents:
        print("\n✅ Nothing to do — all PDFs are already in the vector store.")
        return vector_store

    # 4. Chunk the new documents
    chunks = split_documents(new_documents)

    if not chunks:
        print("\n✅ No chunks produced — nothing added.")
        return vector_store

    # 5. Generate embeddings only for new chunks
    embedding_manager = EmbeddingManager()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    # 6. Store in vector DB
    vector_store.add_documents(chunks, embeddings)

    print(f"\n✅ Ingestion complete. Total docs in store: {vector_store.collection.count()}")
    return vector_store


if __name__ == "__main__":
    ingest_pdfs()