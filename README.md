# basic_rag - Retrieval-Augmented Generation Application

## Overview

This is a RAG (Retrieval-Augmented Generation) application that combines local LLMs with a vector search system to answer questions based on a knowledge base. The application is built with:

- **FastAPI** - Web API framework
- **LlamaIndex** - RAG pipeline and vector store management
- **LangGraph** - Workflow orchestration with retry logic
- **Ollama** - Local LLM and embedding models
- **Chroma** - Vector database

## Architecture

The system works in the following flow:

1. **Document Indexing**: PDF documents from the `docs/` directory are loaded and converted into vector embeddings
2. **Query Processing**: User questions are embedded and searched against the vector store to find relevant context
3. **Answer Generation**: The LLM generates an answer based on the retrieved context
4. **Quality Evaluation**: A LangGraph workflow evaluates if the answer is satisfactory; if not, it retries up to 2 times

### Key Components

- **`main.py`** - FastAPI application with `/health` and `/query` endpoints
- **`rag.py`** - LlamaIndex configuration (LLM, embeddings, vector store, index building)
- **`rag_workflow.py`** - LangGraph state machine with retrieval and evaluation nodes
- **`Modelfile`** - Ollama model configuration (qwen3.5:2b-q4_K_M)
- **`docs/EST 1832.pdf`** - The current knowledge base document

## Prerequisites

### Ollama Setup

Ensure Ollama is installed and running at `http://localhost:11434`. You need two models:

```bash
# Pull the LLM model
ollama pull qwen3.5:2b-q4_K_M

# Pull the embedding model  
ollama pull qwen3-embedding:0.6b-q8_0
```

### Python Dependencies

The project uses a virtual environment at `./venv/`. Activate it and install dependencies:

```bash
source venv/bin/activate
pip install -e .
# Or install from requirements if available
```

## Installation

1. **Clone or navigate to the project**:

   ```bash
   cd basic_rag
   ```

2. **Activate the virtual environment**:

   ```bash
   source venv/bin/activate
   ```

3. **Start Ollama daemon**:

   ```bash
   ollama serve
   ```

4. **Pull required models**:

   ```bash
   ollama pull qwen3.5:2b-q4_K_M
   ollama pull qwen3-embedding:0.6b-q8_0
   ```

## Usage

### Start the API Server

```bash
source venv/bin/activate
uvicorn main:app --reload
```

The server will be available at `http://127.0.0.1:8000`.

### API Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

Response: `{"status": "ok"}`

#### Query Questions

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "your question here"}'
```

Response: A JSON object containing:
- `answer`: The generated answer from the LLM
- `attempts`: Number of retry attempts (max 2)

### Programmatic Usage

```python
from main import app
import httpx

# Health check
response = httpx.get("http://127.0.0.1:8000/health")
print(response.json())  # {"status": "ok"}

# Query
response = httpx.post(
    "http://127.0.0.1:8000/query",
    json={"question": "What is this document about?"}
)
print(response.json())
```

## Configuration

### Modelfile

The `Modelfile` specifies the Ollama model parameters:

```
FROM qwen3.5:2b-q4_K_M
PARAMETER num_ctx 8192
```

### Vector Store

- Uses Chroma in-memory by default (data lost on restart)
- To persist, change `rag.py` to use `chromadb.PersistentClient(path="./chroma_db")`
- Documents are read from `./docs/` directory

### Workflow Parameters

The LangGraph workflow in `rag_workflow.py` has these configurable parameters:

- **Similarity top-k**: 3 (number of relevant documents retrieved)
- **Answer length threshold**: 50 characters (if answer is shorter, retry)
- **Retry logic**: Checks for "I don't know" or "no information" in answer
- **Max retries**: 2 attempts

## Project Structure

```
basic_rag/
├── main.py              # FastAPI application
├── rag.py               # LlamaIndex configuration
├── rag_workflow.py      # LangGraph workflow
├── Modelfile            # Ollama model config
├── opencode.json        # OpenCode configuration
├── docs/
│   └── EST 1832.pdf     # Knowledge base document
├── venv/                # Python virtual environment
└── responses.odt        # Sample document
```

## Development

### Adding More Documents

Place additional PDF files in the `docs/` directory and rebuild the index:

```python
from rag import build_index
index = build_index()  # Can specify path: build_index("./new_docs")
```

### Modifying the Workflow

The LangGraph workflow can be modified in `rag_workflow.py`:
- Change the number of retries by modifying the `attempts < 2` condition
- Adjust the similarity threshold by changing `similarity_top_k` in `get_query_engine()`
- Modify the evaluation heuristic in `evaluate_node()`

## Troubleshooting

### Common Issues

1. **Ollama not running**: Ensure `ollama serve` is running before starting the app
2. **Model not found**: Pull the required models with `ollama pull`
3. **Memory errors**: The models use Q4_K_M quantization; reduce context window if needed
4. **Empty answers**: The workflow retries if answer < 50 chars or contains "I don't know"
5. **Connection refused on port 11434**: Check that Ollama is running and accessible

### Debug Mode

Run with verbose output:

```bash
uvicorn main:app --reload --log-level debug
```