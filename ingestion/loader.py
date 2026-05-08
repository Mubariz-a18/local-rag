"""
Document Loader Module
Loads documents from various sources (txt, pdf, etc.)
"""

from pathlib import Path
from typing import List


class DocumentLoader:
    """Load documents from files."""
    
    def __init__(self, data_path: str = "./data"):
        """Initialize the document loader.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = Path(data_path)
    
    def load_text_file(self, filename: str) -> str:
        """Load a text file.
        
        Args:
            filename: Name of the text file
            
        Returns:
            Content of the file as string
        """
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_all_files(self) -> List[dict]:
        """Load all text files from data directory.
        
        Returns:
            List of documents with metadata
        """
        documents = []
        for file_path in self.data_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "filename": file_path.name,
                    "content": content
                })
        return documents


def load_documents(data_path: str = "./data") -> List[dict]:
    """Load all documents from the data directory.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        List of documents
    """
    loader = DocumentLoader(data_path)
    return loader.load_all_files()
