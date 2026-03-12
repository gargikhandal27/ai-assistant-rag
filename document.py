# pyright: reportMissingImports=false
#!/usr/bin/env python
# coding: utf-8

# ### doc structure

# In[1]:


from langchain_core.documents import Document 


# In[2]:


"""doc=Document(
    page_content="this is the main text content I am using to creat RAG",
    metadata={
        "source":"example.txt",
        "number of pages":1,
        "author":"Gargi Khandal",
        "data_created":"28-02-2026"
    }
    ) """
"""doc"""


# In[3]:


from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

"""dir_loader = DirectoryLoader(
    "data/pdf",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False
)

pdf_documents = dir_loader.load()"""

"""pdf_documents"""


# In[4]:


#type(pdf_documents[0])


# In[5]:


import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


# In[9]:


### Read all the pdf's inside the directory
def process_all_pdfs(pdf_directory):
    """Process all PDF files in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)

    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()

            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

# Process all PDFs in the data directory
all_pdf_documents = process_all_pdfs("data/pdf")


# In[10]:


all_pdf_documents


# In[11]:


### Text splitting get into chunks

def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs


# In[12]:


chunks=split_documents(all_pdf_documents)
chunks


# ### embedding and vectordb
# 

# In[13]:


import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb 
from chromadb.config import Settings
import uuid 
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


# In[14]:


class EmbeddingManager:
    """Handels document embedding generation using sentenceTransformer """
    def __init__(self,model_name:str="all-MiniLM-L6-v2"):
        """Initialize the embedding manager 
        Args:
        model_name: HuggingFace model ame for sentence embedding"""
        self.model_name=model_name
        self.model=None
        self._load_model()

    def _load_model(self):
        """Load the sentence Transformer"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model=SentenceTransformer(self.model_name)
            print(f"Model loaded sucessfully. Embedding dimension:{self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self,texts:List[str]) ->np.ndarray:
        """Generate embeddingsfor a lists of text
        Args:
            texts: List of text strings to embed

        Returns: 
            numpy array of embeddings with shape (len(texts),embedding_dim)
            """
        if not self.model:
            raise ValueError("Model not loaded")

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings=self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def get_embedding_dimension(self) -> int:
        """Get the embedded dimension of the model"""
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.get_embedding_dimension()

## initialize the embedding manager 
embedding_manager= EmbeddingManager()
embedding_manager


# ### VectorStore

# In[15]:


class VectorStore:
    """manage document embeddings in a chromadb vector store"""
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str="data/vector_store"):
        """Initialize the vector store
        Args:
        collection_name: Name of chromadb collection
        persist_directory: Directory to persist the vector store
        """
        self.collection_name=collection_name
        self.persist_directory=persist_directory
        self.client=None
        self.collection=None
        self._initialize_store()

    def _initialize_store(self):
        """Initialze chormaDB client and collection """
        try:
            #create presistent chromadb client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client=chromadb.PersistentClient(path=self.persist_directory)

            #get or create collection
            self.collection=self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description":"PDF documents embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise 

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and there embeddings to the vector store
        Args:
        documents: List of langchain documents
        embeddings: Corresponding embeddings to the documents
        """ 
        if len(documents)!=len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        print(f"Adding {len(documents)} documents to vector store...")

        #Prepare data for chroma db
        ids=[]
        metadatas=[]
        documents_text=[]
        embeddings_list=[]

        for i,(doc,embedding) in enumerate(zip(documents,embeddings)):
            #generate unique id
            doc_id=f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            #prepare metadata
            metadata=dict(doc.metadata)
            metadata['doc_index']=i
            metadata['content_length']=len(doc.page_content)
            metadatas.append(metadata)

            #document content
            documents_text.append(doc.page_content)

            #embedding
            embeddings_list.append(embedding.tolist())

        #add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text               

            )
            print(f"sucessfully added {len(documents)}")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e} ")
            raise

vectorstore=VectorStore()
vectorstore








# In[16]:


chunks 


# In[17]:


### Convert the text to embeddings
texts=[doc.page_content for doc in chunks]

## Generate the Embeddings

if vectorstore.collection.count() == 0:
    embeddings = embedding_manager.generate_embeddings(texts)
    vectorstore.add_documents(chunks, embeddings)
else:
    print("Vector store already contains embeddings. Skipping embedding step.")


# ### RAG RETRIVER PIPELINE

# In[18]:


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever

        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

rag_retriever=RAGRetriever(vectorstore,embedding_manager)


# In[19]:


rag_retriever


# In[20]:


"""rag_retriever.retrieve("What is a Partial differential equation")"""


# ### integrate vectordb with llm output

# In[21]:


##simple RAG pipeline using gemine LLM model
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

print(os.getenv("GOOGLE_API_KEY"))


# In[22]:


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


# In[23]:


class GeminiLLM:
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None):
        """
        Initialize Gemini LLM

        Args:
            model_name: Gemini model name
            api_key: Google API key (or set GOOGLE_API_KEY environment variable)
        """

        self.model_name=model_name
        self.api_key=api_key or os.environ.get("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.1
        )

        print(f"Initialized Gemini LLM with model: {self.model_name}")

    def generate_response(self, query: str, context: str) -> str:
        """
        Generate response using retrieved context
        """

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer:
"""
        )

        formatted_prompt = prompt_template.format(
            context=context,
            question=query
        )

        try:
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_response_simple(self, query: str, context: str) -> str:
        """
        Simple response generation
        """

        simple_prompt = f"""
Based on this context:
{context}

Question: {query}

Answer:
"""

        try:
            messages = [HumanMessage(content=simple_prompt)]
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            return f"Error: {str(e)}"


# In[24]:


# Initialize Gemini LLM (you'll need to set GOOGLE_API_KEY environment variable)

try:
    gemini_llm = GeminiLLM(api_key=os.getenv("GOOGLE_API_KEY"))
    print("Gemini LLM initialized successfully!")

except ValueError as e:
    print(f"Warning: {e}")
    print("Please set your GOOGLE_API_KEY environment variable to use the LLM.")
    gemini_llm = None


# In[25]:


### get the context from the retriever and pass it to the LLM



# Retrieve relevant documents
"""results = rag_retriever.retrieve("what is partial differential equation")"""

# Extract context
"""context = "\n\n".join([doc["content"] for doc in results])"""

# Generate answer
"""answer = gemini_llm.generate_response("what is partial differential equation", context)"""

"""print("Answer:", answer)
"""

# In[26]:


### Simple RAG pipeline with Gemini LLM

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

### Initialize the Gemini LLM (set GOOGLE_API_KEY in environment)
google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0.1
)


## 2. Simple RAG function: retrieve context + generate response
def rag_simple(query, retriever, llm, top_k=3):

    # retrieve context
    results = retriever.retrieve(query, top_k=top_k)

    context = "\n\n".join([doc['content'] for doc in results]) if results else ""

    if not context:
        return "No relevant context found to answer the question."

    # generate answer using Gemini
    prompt = f"""Use the following context to answer the question concisely.

Context:
{context}

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


# In[27]:


"""answer=rag_simple("What is partial differential equation?",rag_retriever,llm)
print(answer)"""


# In[28]:


# --- Enhanced RAG Pipeline Features ---
def rag_advanced(query, retriever, llm, top_k=5, min_score=0.0, return_context=False):
    """
    RAG pipeline with extra features:
    - Returns answer, sources, confidence score, and optionally full context.
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}

    # Prepare context and sources
    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]
    confidence = max([doc['similarity_score'] for doc in results])

    # Generate answer
    prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
    response = llm.invoke([prompt.format(context=context, query=query)])

    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }
    if return_context:
        output['context'] = context
    return output

# Example usage:
"""result = rag_advanced("What is a partial differential equation?", rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)
print("Answer:", result['answer'])
print("Sources:", result['sources'])
print("Confidence:", result['confidence'])
print("Context Preview:", result['context'][:300])
"""

# In[29]:


# --- Advanced RAG Pipeline: Streaming, Citations, History, Summarization ---
from typing import List, Dict, Any
import time

class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []  # Store query history

    def query(self, question: str, top_k: int = 5, min_score: float = 0.0, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]
            # Streaming answer simulation
            prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            response = self.llm.invoke([prompt.format(context=context, question=question)])
            answer = response.content

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }

# Example usage:
adv_rag = AdvancedRAGPipeline(rag_retriever, llm)
"""result = adv_rag.query("define partial differential equation?", top_k=3, min_score=0.0, stream=True, summarize=True)
print("\nFinal Answer:", result['answer'])
print("Summary:", result['summary'])
print("History:", result['history'][-1])
"""
