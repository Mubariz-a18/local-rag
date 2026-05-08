# RAG-AI: Retrieval-Augmented Generation System

A local, privacy-focused RAG (Retrieval-Augmented Generation) system that combines document retrieval with local LLM inference using Ollama and FAISS vector store.

## Features

- **Document Ingestion**: Load and process documents from local files
- **Vector Embeddings**: Convert documents to embeddings using OpenAI's API
- **FAISS Vector Store**: Efficient similarity search with persistent storage
- **Local LLM**: Generate responses using Ollama (runs locally, no cloud APIs)
- **RAG Pipeline**: Retrieve relevant context and generate informed responses

## Project Structure

```
RAG-AI/
├── app.py                 # Main application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env_config            # API keys and environment variables
├── data/
│   ├── data.txt           # Sample documents to ingest
│   └── vector_store/      # FAISS index and embeddings storage
├── generation/
│   └── llm.py             # LLM generation module (Ollama)
├── ingestion/
│   ├── loader.py          # Document loader
│   ├── embeddings.py      # Embedding generation
│   └── splitter.py        # Document chunking
└── retrieval/
    └── retriever.py       # Similarity search module
```

## Setup

### 1. Install Dependencies

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

### 2. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from: https://ollama.ai
```

### 3. Configure API Keys

Edit `.env_config`:

```
OPENAI_API_KEY=your-openai-api-key  # For embeddings
HF_API_TOKEN=your-hugging-face-token  # Optional backup
```

### 4. Start Ollama

In a new terminal:

```bash
ollama serve
```

Download the Mistral model (first time only):

```bash
ollama pull mistral
```

## Usage

### Ingest Documents

```bash
python app.py ingest data/data.txt
```

This will:
1. Load documents from `data/data.txt`
2. Split them into chunks
3. Generate embeddings using OpenAI
4. Store in FAISS vector store

### Query Documents

```bash
python app.py query "What are the risk factors mentioned?"
```

This will:
1. Retrieve relevant document chunks from FAISS
2. Send them to Ollama with your query
3. Return an informed response based on the documents

## Configuration

Edit `config.py` to customize:

- `CHUNK_SIZE`: Document chunk size (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `TEMPERATURE`: LLM generation temperature (default: 0.7)
- `MAX_TOKENS`: Max tokens to generate (default: 200)
- `OLLAMA_MODEL`: Local model to use (default: mistral)

## Troubleshooting

### "Ollama not running"
Start Ollama in another terminal: `ollama serve`

### "Model not found"
Pull the model: `ollama pull mistral`

### "OpenAI API key invalid"
Update `OPENAI_API_KEY` in `.env_config`

### Poor response quality
- Increase `MAX_TOKENS` in `config.py`
- Use a larger Ollama model: `ollama pull neural-chat`
- Ensure document chunks are relevant to your queries

## Supported Ollama Models

- `mistral` (7B, recommended)
- `neural-chat` (7B, better instruction following)
- `llama2` (7B)
- `dolphin-mixtral` (8x7B, more capable but slower)

Pull any model with: `ollama pull <model-name>`

## Technologies

- **Embeddings**: OpenAI
- **Vector Store**: FAISS
- **LLM**: Ollama (local)
- **Framework**: Python 3.9+

## License

MIT

## Contributing

Feel free to fork, modify, and improve!
