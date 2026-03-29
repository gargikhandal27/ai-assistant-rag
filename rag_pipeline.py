from core.pdf_loader import process_all_pdfs, split_documents
from core.embedding_manager import EmbeddingManager
from core.vector_store import VectorStore
from core.retriever import RAGRetriever
from core.llm_handler import GeminiLLM, AdvancedRAGPipeline


def initialize_rag():
    """Initialize full RAG pipeline"""

    # Load all PDFs
    documents = process_all_pdfs("data/pdf")

    # Split into chunks
    chunks = split_documents(documents)

    # Initialize embedding model
    embedding_manager = EmbeddingManager()

    # Initialize vector store
    vectorstore = VectorStore()

    # Check already indexed PDFs
    existing_files = vectorstore.get_existing_files()

    # Only add new PDFs
    new_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("source_file")
        not in existing_files
    ]

    if new_chunks:
        print(
            f"Adding {len(new_chunks)} new chunks"
        )

        texts = [
            doc.page_content
            for doc in new_chunks
        ]

        embeddings = (
            embedding_manager
            .generate_embeddings(texts)
        )

        vectorstore.add_documents(
            new_chunks,
            embeddings
        )

    else:
        print(
            "All PDFs already indexed."
        )

    # Retriever
    retriever = RAGRetriever(
        vectorstore,
        embedding_manager
    )

    # LLM
    llm_handler = GeminiLLM()

    # Full advanced pipeline
    return AdvancedRAGPipeline(
        retriever,
        llm_handler
    )


# Initialize once
rag_pipeline = initialize_rag()


def run_rag(query: str):
    """Run query on existing pipeline"""

    result = rag_pipeline.query(query)

    return result["answer"]