"""
Embeddings Module
Create embeddings using OpenAI and store them in FAISS vector database
"""

from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os


class EmbeddingStore:
    """Store and manage embeddings using FAISS."""
    
    def __init__(self, store_path: str = "./data/vector_store"):
        """Initialize the embedding store.
        
        Args:
            store_path: Path to store FAISS index
        """
        self.store_path = store_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
    
    def create_embeddings(self, chunks: List[dict]) -> FAISS:
        """Create embeddings for text chunks using OpenAI and store in FAISS.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            FAISS vector store
        """
        # Convert chunks to LangChain Document format
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    "filename": chunk["filename"],
                    "chunk_id": chunk["chunk_id"]
                }
            )
            documents.append(doc)
        
        # Create FAISS vector store from documents
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        return self.vector_store
    
    def save_embeddings(self, force: bool = False) -> None:
        """Save FAISS index to disk.
        
        Args:
            force: Force save even if index exists
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create embeddings first.")
        
        os.makedirs(self.store_path, exist_ok=True)
        self.vector_store.save_local(self.store_path)
        print(f"Vector store saved to {self.store_path}")
    
    def load_embeddings(self) -> FAISS:
        """Load FAISS index from disk.
        
        Returns:
            FAISS vector store
        """
        try:
            self.vector_store = FAISS.load_local(
                self.store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return self.vector_store
        except Exception as e:
            print(f"Could not load vector store: {e}")
            return None
    
    def get_vector_store(self) -> FAISS:
        """Get the current vector store.
        
        Returns:
            FAISS vector store
        """
        return self.vector_store


def create_and_store_embeddings(chunks: List[dict], 
                               store_path: str = "./data/vector_store") -> FAISS:
    """Create embeddings and store them in FAISS.
    
    Args:
        chunks: List of text chunks
        store_path: Path to store embeddings
        
    Returns:
        FAISS vector store
    """
    store = EmbeddingStore(store_path)
    vector_store = store.create_embeddings(chunks)
    store.save_embeddings()
    return vector_store
