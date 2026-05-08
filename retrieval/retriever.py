"""
Retriever Module
Retrieve relevant documents using FAISS vector store
"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


class Retriever:
    """Retrieve relevant chunks using FAISS similarity search."""
    
    def __init__(self, store_path: str = "./data/vector_store"):
        """Initialize the retriever.
        
        Args:
            store_path: Path to FAISS vector store
        """
        self.store_path = store_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = self._load_vector_store()
    
    def _load_vector_store(self) -> FAISS:
        """Load FAISS vector store from disk.
        
        Returns:
            FAISS vector store or None if not found
        """
        try:
            vector_store = FAISS.load_local(
                self.store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            print(f"Could not load vector store: {e}")
            return None
    
    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        """Retrieve top-k relevant chunks using FAISS.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if self.vector_store is None:
            return []
        
        try:
            # Use FAISS similarity search with score
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            # Format results
            chunks = []
            for doc, score in results:
                chunks.append({
                    "content": doc.page_content,
                    "filename": doc.metadata.get("filename", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "similarity": 1 - score  # Convert distance to similarity
                })
            
            return chunks
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []


def retrieve_relevant_chunks(query: str, top_k: int = 3,
                           store_path: str = "./data/vector_store") -> List[dict]:
    """Retrieve relevant chunks for a query using FAISS.
    
    Args:
        query: Query string
        top_k: Number of results to return
        store_path: Path to FAISS vector store
        
    Returns:
        List of relevant chunks
    """
    retriever = Retriever(store_path)
    return retriever.retrieve(query, top_k)
