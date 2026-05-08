"""
Document Splitter Module
Chunks documents into smaller pieces for embedding
"""

from typing import List


class DocumentSplitter:
    """Split documents into chunks."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the splitter.
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def split_documents(self, documents: List[dict]) -> List[dict]:
        """Split multiple documents into chunks.
        
        Args:
            documents: List of documents with content
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc["content"])
            for i, chunk in enumerate(text_chunks):
                chunks.append({
                    "filename": doc["filename"],
                    "chunk_id": i,
                    "content": chunk
                })
        
        return chunks


def split_documents(documents: List[dict], chunk_size: int = 500, 
                   chunk_overlap: int = 50) -> List[dict]:
    """Split documents into chunks.
    
    Args:
        documents: List of documents
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunks
    """
    splitter = DocumentSplitter(chunk_size, chunk_overlap)
    return splitter.split_documents(documents)
