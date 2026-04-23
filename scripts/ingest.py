"""
ingest.py — Run this script to load new PDFs into the vector store.
Already-indexed files are automatically skipped.
"""

from core.pdf_loader import process_all_pdfs, split_documents
from core.embedding_manager import EmbeddingManager
from core.vector_store import VectorStore

PDF_DIRECTORY = "data/pdf"      #path where pdfs exist 


def ingest_pdfs():
    #creates vector db object (folder created, db loaded, collection is ready)
    vector_store = VectorStore(
        collection_name="pdf_documents",
        persist_directory="data/vector_store"
    )

    
    existing_files = vector_store.get_existing_files()      #returns set of already stored pdfs 
    
    #check if set is not empty 
    if existing_files:
        print(f"\nAlready indexed: {existing_files}\n")
    else:
        print("\nNo files indexed yet — fresh start.\n")

    #loads only those pdfs which are not in existing files 
    new_documents = process_all_pdfs(PDF_DIRECTORY, existing_files=existing_files)

    #if list is empty stops fun early and returns vector store 
    if not new_documents:
        print("\n✅ Nothing to do — all PDFs are already in the vector store.")
        return vector_store

    #chunking
    chunks = split_documents(new_documents)

    #checks if splitting fails 
    if not chunks:
        print("\nNo chunks produced — nothing added.")
        return vector_store

    #Generate embeddings only for new chunks
    embedding_manager = EmbeddingManager()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    #Store in vector DB
    vector_store.add_documents(chunks, embeddings)

    print(f"\nIngestion complete. Total docs in store: {vector_store.collection.count()}")
    return vector_store

#checks if file run directly then calss function and starts entire pipeline 
if __name__ == "__main__":
    ingest_pdfs()