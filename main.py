import os
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import initialize_rag, run_rag
from core.llm_handler import PROVIDERS, PROVIDER_MODELS, PROVIDER_ENV_KEYS, PROVIDER_LINKS

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="📚",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Assistant")
    st.markdown("---")

    st.subheader("🤖 LLM Settings")

    provider = st.selectbox(
        "Provider",
        options=list(PROVIDERS.keys()),
        index=0,
    )

    model = st.selectbox(
        "Model",
        options=PROVIDER_MODELS[provider],
        index=0,
    )
    apply = st.button(
    "✅ Apply & Initialize",
    use_container_width=True
    )

    if apply:
        with st.spinner(f"Initializing {provider} — {model}…"):
            try:
                initialize_rag(provider=provider, model=model)
                st.session_state["llm_ready"] = True
                st.session_state["active_provider"] = provider
                st.session_state["active_model"] = model
                st.success(f"✅ Ready! Using **{model}**")
            except Exception as e:
                st.session_state["llm_ready"] = False
                st.error(f"Error: {e}")

    # Active model badge
    if st.session_state.get("llm_ready"):
        st.markdown(
            f"**Active:** `{st.session_state['active_model']}`  \n"
            f"**Provider:** {st.session_state['active_provider']}"
        )

    st.markdown("---")
    st.subheader("ℹ️ About")
    st.write(
        "RAG (Retrieval-Augmented Generation) answers your questions "
        "using only the documents you've loaded."
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📚 AI Document Chatbot")

if not st.session_state.get("llm_ready"):
    st.info("👈 Select a provider and model, then click **Apply & Initialize** to start.")

    
else:
    st.caption(
        f"Powered by **{st.session_state['active_model']}** "
        f"({st.session_state['active_provider']})"
    )

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input(
    "Ask a question about your documents…",
    disabled=not st.session_state.get("llm_ready")
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            answer = run_rag(query)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})