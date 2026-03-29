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
