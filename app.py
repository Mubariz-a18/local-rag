"""
Main Application File
Orchestrates the RAG pipeline
"""

import sys
from config import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_STORE_PATH
from ingestion.loader import load_documents
from ingestion.splitter import split_documents
from ingestion.embeddings import create_and_store_embeddings
from retrieval.retriever import retrieve_relevant_chunks
from generation.llm import generate_response


def ingest_data():
    """Ingest documents, chunk them, and create embeddings."""
    print("Loading documents...")
    documents = load_documents("./data")
    
    if not documents:
        print("No documents found in data/ directory")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    print("Splitting documents into chunks...")
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating embeddings...")
    vector_store = create_and_store_embeddings(chunks, VECTOR_STORE_PATH)
    print(f"Stored embeddings in FAISS vector store")
    print("Ingestion complete!\n")


def query_rag(query: str):
    """Query the RAG system.
    
    Args:
        query: User query
    """
    print(f"Query: {query}\n")
    
    # Retrieve relevant chunks
    print("Retrieving relevant documents...")
    context_chunks = retrieve_relevant_chunks(query, top_k=3, store_path=VECTOR_STORE_PATH)
    
    if not context_chunks:
        print("No relevant documents found")
        return
    
    print(f"Found {len(context_chunks)} relevant chunks:\n")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"[Chunk {i}] (Similarity: {chunk.get('similarity', 'N/A'):.2f})")
        print(f"Content: {chunk['content'][:100]}...\n")
    
    # Generate response
    print("Generating response...")
    response = generate_response(query, context_chunks)
    print(f"Response: {response}\n")


def main():
    """Main function."""
    print("=" * 50)
    print("RAG Application")
    print("=" * 50 + "\n")
    
    # Ingest data
    ingest_data()
    
    # Example queries
    example_queries = [
        # "What are the major risk factors for heart disease?",
        # "How can I prevent cardiovascular disease?",
        "What are the early warning signs of a heart attack?"
    ]
    
    print("Processing example queries...\n")
    for query in example_queries:
        query_rag(query)
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
