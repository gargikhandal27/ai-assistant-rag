"""
rag_pipeline.py

Main entry point for the entire RAG system.

What changed from old version:
  OLD → used AdvancedRAGPipeline + FallbackLLM (your code decided everything)
  NEW → uses AgenticRAGPipeline (AI decides model, tools, web search, email)

Everything else (PDF loading, vector store, retriever) stays exactly the same.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from core.pdf_loader import process_all_pdfs, split_documents
from core.embedding_manager import EmbeddingManager
from core.vector_store import VectorStore
from core.retriever import RAGRetriever
from llms.llm_handler import build_available_llms          # ← NEW: loads all LLMs at once
from agent.agent import AgenticRAGPipeline                  # ← NEW: agentic pipeline


# ── Global singletons (created once, reused forever) ─────────────────────────
_pipeline:         AgenticRAGPipeline | None = None
_retriever:        RAGRetriever        | None = None
_embedding_manager: EmbeddingManager   | None = None
_vectorstore:      VectorStore         | None = None


# ── Internal helper ───────────────────────────────────────────────────────────

def _ensure_store():
    """
    Initialize embedding manager, vector store, and retriever
    if they haven't been created yet.
    This runs only ONCE on first use.
    """
    global _embedding_manager, _vectorstore, _retriever

    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()

    if _vectorstore is None:
        _vectorstore = VectorStore(
            collection_name="pdf_documents",
            persist_directory="data/vector_store",
        )

    if _retriever is None:
        _retriever = RAGRetriever(_vectorstore, _embedding_manager)


# ── Main initializer ──────────────────────────────────────────────────────────

def initialize_rag() -> AgenticRAGPipeline:
    """
    Build the full Agentic RAG pipeline.

    What happens here:
      1. Load embedding model + vector store (first time only)
      2. Ingest any new PDFs from data/pdf/ folder
      3. Load all available LLMs (Groq, Gemini, OpenRouter)
      4. Create AgenticRAGPipeline — AI will now decide everything

    API keys are read automatically from your .env file.
    Call this once at app startup.
    """
    global _pipeline, _retriever

    # ── Step 1: First-time store setup ───────────────────────────────────
    if _retriever is None:
        _ensure_store()

        # Load only NEW pdfs (already indexed ones are skipped)
        existing_files = _vectorstore.get_existing_files()
        new_docs       = process_all_pdfs("data/pdf", existing_files=existing_files)
        chunks         = split_documents(new_docs)

        # Filter to only truly new chunks
        new_chunks = [
            c for c in chunks
            if c.metadata.get("source_file") not in existing_files
        ]

        if new_chunks:
            print(f"Adding {len(new_chunks)} new chunks to vector store...")
            texts      = [c.page_content for c in new_chunks]
            embeddings = _embedding_manager.generate_embeddings(texts)
            _vectorstore.add_documents(new_chunks, embeddings)
        else:
            print("All PDFs already indexed. Nothing new to add.")

        _retriever = RAGRetriever(_vectorstore, _embedding_manager)

    # ── Step 2: Load all available LLMs ──────────────────────────────────
    # build_available_llms() tries Groq, Gemini, OpenRouter
    # Only ones with valid API keys in .env will load
    # If a key is missing, that model is simply skipped (no crash)
    available_llms = build_available_llms()

    # ── Step 3: Create Agentic Pipeline ──────────────────────────────────
    # From now on, the AI decides:
    #   - which model to use (RouterAgent)
    #   - whether to search web (tool decision)
    #   - whether to send email (tool decision)
    _pipeline = AgenticRAGPipeline(_retriever, available_llms)

    return _pipeline


# ── PDF Upload Handler ────────────────────────────────────────────────────────

def ingest_uploaded_pdf(pdf_path: str) -> int:
    """
    Ingest a single newly uploaded PDF into the vector store.
    Called when user uploads a PDF through the UI.
    Returns number of chunks added (0 if already indexed).
    """
    _ensure_store()

    pdf_file       = Path(pdf_path)
    existing_files = _vectorstore.get_existing_files()

    # Skip if already in vector store
    if pdf_file.name in existing_files:
        print(f"Already indexed: {pdf_file.name}")
        return 0

    # Load pages from PDF
    loader    = PyPDFLoader(str(pdf_file))
    documents = loader.load()
    for doc in documents:
        doc.metadata["source_file"] = pdf_file.name
        doc.metadata["file_type"]   = "pdf"

    # Split into chunks
    chunks = split_documents(documents)
    if not chunks:
        return 0

    # Embed and store
    texts      = [c.page_content for c in chunks]
    embeddings = _embedding_manager.generate_embeddings(texts)
    _vectorstore.add_documents(chunks, embeddings)

    # Make sure retriever sees the new data
    if _retriever is not None:
        _retriever.vector_store = _vectorstore

    print(f"Ingested {pdf_file.name}: {len(chunks)} chunks added")
    return len(chunks)


# ── Query Runner ──────────────────────────────────────────────────────────────

def run_rag(query: str) -> dict:
    """
    Run a query through the agentic pipeline.

    Returns the full result dict which includes:
      - answer          → the final answer text with citations
      - active_model    → which model actually answered
      - routing_decision → why that model was chosen
      - tool_decision   → which tools were used and why
      - reasoning_trail → every step the agent took
      - sources         → PDF pages used
      - web_results     → web search results (if used)
      - email_result    → email status (if email was requested)
    """
    if _pipeline is None:
        return {
            "answer": "Pipeline not initialized. Please call initialize_rag() first.",
            "active_model":     "none",
            "routing_decision": {},
            "tool_decision":    {},
            "reasoning_trail":  [],
            "sources":          [],
        }

    return _pipeline.query(query)