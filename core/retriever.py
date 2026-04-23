from typing import List, Dict, Any


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store, embedding_manager):        #define constructor to define dependencies 
        self.vector_store = vector_store        #vectordb refrence
        self.embedding_manager = embedding_manager          #embedding generator refrence 

    def retrieve(self,query: str,top_k: int = 5,score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        """
        #kist specify return type

        print(f"Retrieving documents for query: '{query}'")     #display input query 
        print(
            f"Top K: {top_k}, Score threshold: {score_threshold}"       #display retrivel parameters 
        )

        try:
            # convert query string into its embedding
            query_embedding = (
                self.embedding_manager
                .generate_embeddings([query])[0]
            )

            # calls vector DB search
            results = self.vector_store.collection.query(
                query_embeddings=[
                    query_embedding.tolist()
                ],
                n_results=top_k     #limit number of results return 
            )

            retrieved_docs = []     #empty list to store final result 

            if (            #check if db returned any document 
                results["documents"]
                and results["documents"][0]
            ):

                documents = results["documents"][0]     #list of retrived text 
                metadatas = results["metadatas"][0]      #retrive metadata
                distances = results["distances"][0]       #distance score
                ids = results["ids"][0]     #documents id 

                for i, (            #Iterates over each retrived results 
                    doc_id,
                    document,
                    metadata,
                    distance
                ) in enumerate(
                    zip(
                        ids,
                        documents,
                        metadatas,
                        distances
                    )
                ):

                    similarity_score = 1 - distance     #convert distance into similarity score 

                    if (
                        similarity_score        #filter result based on threshold 
                        >= score_threshold
                    ):
                        retrieved_docs.append(      #add selected result to list 
                            {
                                "id": doc_id,       #store document id 
                                "content": document,        #document text 
                                "metadata": metadata,       #metadata
                                "similarity_score": similarity_score,   #similarity value
                                "distance": distance,                   #distance
                                "rank": i + 1           #
                            }
                        )

                print(
                    f"Retrieved {len(retrieved_docs)} documents"            #number of retrived results 
                )

            else:
                print("No documents found")

            return retrieved_docs           #final list of docs 

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []