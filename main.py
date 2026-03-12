import streamlit as st
from rag_pipeline import run_rag

# Page settings
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="📚",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("📚 RAG Assistant")
    st.write("Ask questions about your documents.")
    
    st.markdown("---")
    
    st.write("### About")
    st.write(
        "This system uses Retrieval-Augmented Generation (RAG) "
        "to answer questons from your documents."
    )

    
# Main title
st.title("📚 AI Document Chatbot")

st.write("Ask questions based on the loaded documents.")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
query = st.chat_input("Ask a question about your documents...")

if query:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):

            # Replace this with your pipeline
            answer = run_rag(query)

            st.write(answer)

    # Save response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )