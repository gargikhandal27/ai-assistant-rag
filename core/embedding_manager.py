import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingManager:     #handels everything related to embeddings

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"): #initialize embedding manager  starts class and call model loader  
        
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self): #load embedding model(into memory) store it in self model 
        
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)       #load models from huggingface
            print(
                f"Model loaded successfully. "
                f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}"     #gets vector size just for printing
            )
        except Exception as e:
            print(f"Error loading model '{self.model_name}': {e}")
            raise       #stop program 

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:          #take list of texts genrates there vector and return in numpy array

        if not self.model:      #check model exist or not 
            raise ValueError("Model not loaded.")

        print(f"Generating embeddings for {len(texts)} text(s)...")
        embeddings = self.model.encode(texts, show_progress_bar=True)       #inputt list and output vectors 
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings       #send vector back to caller 

    def get_embedding_dimension(self) -> int:       #return size of each embedding vector 
        if not self.model:
            raise ValueError("Model not loaded.")
        return self.model.get_sentence_embedding_dimension()