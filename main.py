"""
main.py

Streamlit UI for the Agentic RAG Pipeline.

What changed:
  - Removed FallbackLLM and all manual model-switching logic
  - The AI (RouterAgent) now decides which model to use
  - Shows the agent's full reasoning trail in the UI
  - Shows tool decisions (web search, email) in the UI
  - No manual provider/model selector needed — agent handles it
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _render_result(result: dict):
    """
    Render agent result cleanly under an answer.
    Rule: answer first, metadata only if meaningful, debug hidden by default.
    """
    model_used = result.get("active_model", "unknown")
    routing    = result.get("routing_decision", {})
    tool       = result.get("tool_decision", {})
    sources    = result.get("sources", [])
    web_results = result.get("web_results", [])
    email_result = result.get("email_result")
    trail      = result.get("reasoning_trail", [])

    # ── One compact info line (model + web search badge if used) ──────────
    web_badge  = " · 🌐 web searched" if tool.get("need_web_search") else ""
    model_name = routing.get("model", model_used)
    st.caption(f"Answered by `{model_name}`{web_badge}")

    # ── Email notification (only when email was actually sent/attempted) ──
    if email_result:
        if email_result.get("success"):
            st.success(f"📧 {email_result['message']}")
        else:
            st.warning(f"📧 Email failed: {email_result['message']}")

    # ── Web results (only shown when web search actually ran) ─────────────
    if web_results:
        with st.expander(f"🌐 Web sources ({len(web_results)})", expanded=False):
            for i, r in enumerate(web_results, 1):
                st.markdown(f"**{i}. {r.get('title', 'Result')}**")
                st.markdown(r.get("snippet", ""))
                if r.get("url"):
                    st.caption(r["url"])

    # ── PDF sources (only shown when chunks were actually retrieved) ───────
    if sources:
        with st.expander(f"📄 Sources ({len(sources)} chunks)", expanded=False):
            for s in sources:
                st.markdown(f"**{s['source']}** · page {s['page']} · score `{s['score']:.2f}`")
                st.caption(s["preview"])

    # ── Debug info — collapsed, only for developers who want to dig in ─────
    with st.expander("🔧 Agent debug info", expanded=False):
        col1, col2 = st.columns(2)
        col1.markdown(f"**Model chosen:** `{routing.get('model', '—')}`")
        col2.markdown(f"**Provider:** {routing.get('provider', '—')}")
        st.caption(f"Routing reason: {routing.get('reason', '—')}")
        st.caption(f"Tool reasoning: {tool.get('reasoning', '—')}")
        if trail:
            st.markdown("**Steps taken:**")
            for step in trail:
                st.markdown(f"- {step['step']}")


st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# ── Initialize pipeline once (cached so it survives reruns) ──────────────────
@st.cache_resource(show_spinner="Initializing Agentic RAG Pipeline...")
def get_pipeline():
    from services.rag_pipeline import initialize_rag
    return initialize_rag()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Agentic RAG")
    st.markdown("---")

    st.subheader("ℹ️ How it works")
    st.markdown(
        """
        The AI agent **automatically**:
        - 🧭 Picks the best model for each question
        - 🌐 Searches the web when needed
        - 📄 Retrieves from your PDFs
        - 📧 Sends email if you ask it to

        No manual model selection required!
        """
    )

    st.markdown("---")
    st.subheader("🤖 Available Models")
    st.markdown(
        """
        - **Gemini** → complex reasoning, long docs
        - **Groq** → fast answers, summaries
        - **OpenRouter** → creative, general knowledge
        """
    )

    st.markdown("---")
    st.subheader("📄 Upload PDF")

    uploaded_files = st.file_uploader(
        "Add PDFs to your knowledge base",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Ingest PDFs", use_container_width=True):
            from services.rag_pipeline import ingest_uploaded_pdf
            pdf_dir = "data/pdf"
            os.makedirs(pdf_dir, exist_ok=True)
            results = []
            for uploaded_file in uploaded_files:
                dest_path = os.path.join(pdf_dir, uploaded_file.name)
                if os.path.exists(dest_path):
                    results.append(f"⏭️ **{uploaded_file.name}** — already exists, skipped")
                    continue
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                try:
                    with st.spinner(f"Indexing {uploaded_file.name}..."):
                        chunks_added = ingest_uploaded_pdf(dest_path)
                    results.append(f"✅ **{uploaded_file.name}** — {chunks_added} chunks added")
                except Exception as e:
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    results.append(f"❌ **{uploaded_file.name}** — Error: {e}")
            for r in results:
                st.markdown(r)

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Agentic RAG · RouterAgent picks the model automatically")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🤖 Agentic AI Document Chatbot")
st.caption("The agent decides which model to use, whether to search the web, and more — automatically.")

# Load pipeline
try:
    pipeline = get_pipeline()
    pipeline_ready = True
except Exception as e:
    st.error(f"Failed to initialize pipeline: {e}")
    pipeline_ready = False

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("result"):
            _render_result(message["result"])


# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input(
    "Ask anything — the agent will decide how to best answer...",
    disabled=not pipeline_ready,
)

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent is thinking..."):
            try:
                result = pipeline.query(query)
            except Exception as e:
                result = {
                    "answer": f"Error: {e}",
                    "active_model": "none",
                    "routing_decision": {},
                    "tool_decision": {},
                    "reasoning_trail": [],
                    "sources": [],
                    "web_results": [],
                    "email_result": None,
                }

        answer = result.get("answer", "No answer returned.")
        st.markdown(answer)
        _render_result(result)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", "No answer returned."),
        "result": result,
    })