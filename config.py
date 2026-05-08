"""
Configuration file for RAG application.
Store API keys, model names, and other configurations here.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env_config file
load_dotenv('.env_config')

# OpenAI API Configuration (for embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# Ollama Configuration (for local LLM generation)
# Make sure Ollama is running: ollama serve
OLLAMA_MODEL = "mistral"

# Vector Store Configuration
VECTOR_STORE_PATH = "./data/vector_store"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# LLM Generation Configuration
TEMPERATURE = 0.7
MAX_TOKENS = 200
