from core.pdf_loader import process_all_pdfs, split_documents
from core.embedding_manager import EmbeddingManager
from core.vector_store import VectorStore
from core.retriever import RAGRetriever
from core.llm_handler import AdvancedRAGPipeline, create_llm

_pipeline: AdvancedRAGPipeline | None = None
_retriever: RAGRetriever | None = None


def initialize_rag(provider: str, model: str) -> AdvancedRAGPipeline:
    """
    Build the RAG pipeline.
    First call: loads PDFs, builds vector store, creates retriever.
    Subsequent calls: only swaps the LLM — no PDF reloading.
    API keys are read automatically from .env
    """
    global _pipeline, _retriever

    # ── First-time setup ─────────────────────────────────────────────────────
    if _retriever is None:
        embedding_manager = EmbeddingManager()
        vectorstore = VectorStore()

        existing_files = vectorstore.get_existing_files()
        new_docs = process_all_pdfs("data/pdf", existing_files=existing_files)
        chunks = split_documents(new_docs)

        new_chunks = [
            c for c in chunks
            if c.metadata.get("source_file") not in existing_files
        ]

        if new_chunks:
            print(f"Adding {len(new_chunks)} new chunks to vector store…")
            texts = [c.page_content for c in new_chunks]
            embeddings = embedding_manager.generate_embeddings(texts)
            vectorstore.add_documents(new_chunks, embeddings)
        else:
            print("All PDFs already indexed.")

        _retriever = RAGRetriever(vectorstore, embedding_manager)

    # ── Create / swap LLM (key comes from .env) ───────────────────────────────
    llm = create_llm(provider, model)

    if _pipeline is None:
        _pipeline = AdvancedRAGPipeline(_retriever, llm)
    else:
        _pipeline.swap_llm(llm)

    return _pipeline


def run_rag(query: str) -> str:
    if _pipeline is None:
        return "Pipeline not initialized. Please click **Apply & Initialize** in the sidebar."
    result = _pipeline.query(query)
    return result["answer"]