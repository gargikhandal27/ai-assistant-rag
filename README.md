# AI ASSISTANT – RAG System

This project implements an **AI Assistant based on Retrieval-Augmented Generation (RAG)** that answers questions from pdf.

The system reads documents, converts them into embeddings, stores them in a vector database, retrieves relevant information for a query, and generates answers using a Large Language Model.
Data files are not included in the repository due to size.
Users can add their own PDFs to the /data folder.
---

## Features

* Load and process PDF documents
* Extract and clean text
* Chunk documents for semantic search
* Generate embeddings
* Store embeddings in a vector database
* Retrieve relevant information for user queries
* Generate answers using an LLM

---

## Project Structure

```
AI ASSISTANT
│
├── data
│   ├── pdf/                # Input PDF files
│   ├── text_files/         # Extracted text files
│   └── vector_store/       # Vector database (ignored in git)
│
├── notebook
│   └── document.ipynb      # Experiment notebook
│
├── document.py             # Document processing and text extraction
├── rag_pipeline.py         # RAG pipeline implementation
├── main.py                 # Main script to run the assistant
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
├── README.md               # Project documentation
└── .gitignore
```

---

## How the System Works

The AI assistant follows a **RAG pipeline**:

1. **Document Loading**

   * PDF lecture notes are loaded from the `data/pdf` folder.

2. **Text Extraction**

   * Text is extracted and prepared for processing.

3. **Chunking**

   * Documents are split into smaller chunks for better retrieval.

4. **Embedding Generation**

   * Each chunk is converted into vector embeddings.

5. **Vector Storage**

   * Embeddings are stored in a vector database.

6. **Query Processing**

   * When a question is asked, the system retrieves the most relevant chunks.

7. **Answer Generation**

   * The retrieved context is provided to the LLM to generate an accurate response.

---

## Installation

Clone the repository:

```
git clone https://github.com/gargikhandal27/AI-ASSISTANT.git
cd AI-ASSISTANT
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

Run the main script:

```
python main.py
```

---

## Technologies Used

* Python
* LangChain
* Vector Database
* LLM API
* Document Processing

---

## Future Improvements

* Add a Streamlit chat interface
* Improve retrieval accuracy
* Support more document formats
* Deploy the assistant as a web application

---
## Learning Resources

This project was built while learning about Retrieval-Augmented Generation (RAG) and AI assistants from various online resources including:

- YouTube tutorials on RAG systems
- LangChain documentation
- Vector database documentation
- Research papers and PDE lecture notes

These resources helped in understanding the concepts and implementing the system.

----

## Author

Gargi Khandal
