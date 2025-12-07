# LM Studio RAG with WCAG 2.2

A Retrieval-Augmented Generation (RAG) system that uses LM Studio for local LLM inference and embeddings to query the WCAG 2.2 accessibility guidelines.

## Overview

This project demonstrates how to build a RAG pipeline using:
- **LM Studio** - Local LLM server for running models without cloud dependencies
- **LlamaIndex** - Framework for building LLM applications with document indexing and retrieval
- **WCAG 2.2 PDF** - Web Content Accessibility Guidelines as the knowledge base

The system creates a vector database from the WCAG 2.2 PDF document, enabling semantic search and question-answering with source citations.

## Architecture

```
┌─────────────────┐
│  WCAG 2.2 PDF   │
└────────┬────────┘
         │ Load & Parse
         ▼
┌─────────────────┐
│  Text Chunks    │
└────────┬────────┘
         │ Embed with LM Studio
         ▼
┌─────────────────┐      ┌──────────────────┐
│  Vector Index   │◄─────┤  Query Engine    │
│  (Persisted)    │      └────────┬─────────┘
└─────────────────┘               │
                                  │ Retrieve & Generate
                                  ▼
                          ┌──────────────────┐
                          │  LM Studio LLM   │
                          │  (meta-llama)    │
                          └──────────────────┘
```

## Prerequisites

### 1. Python Environment
- Python 3.11 or higher
- Required packages (install via pip):
```bash
pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai requests
```

### 2. LM Studio Setup
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load the following models in LM Studio:
   - **LLM Model**: `meta-llama-3.1-8b-instruct` (or similar chat model)
   - **Embedding Model**: `text-embedding-nomic-embed-text-v1.5`
3. Start the LM Studio local server:
   - Open LM Studio
   - Go to "Local Server" tab
   - Click "Start Server"
   - Verify it's running on `http://localhost:1234`

### 3. WCAG 2.2 PDF
Place the `wcag-2.2.pdf` file in the project root directory.

## Project Structure

```
lm-rag/
├── README.md                    # This file
├── rag_wcag.py                 # Main RAG application
├── test_lmstudio_llm.py        # Test script for LM Studio connection
├── wcag-2.2.pdf                # WCAG guidelines PDF (required)
└── wcag_index/                 # Persisted vector index (auto-generated)
    ├── docstore.json
    ├── index_store.json
    └── vector_store.json
```

## How It Works

### Custom LM Studio Integration

The code implements two custom classes to integrate LM Studio with LlamaIndex:

#### 1. `LMStudioLLM` (Chat Completions)
- Extends `CustomLLM` to interface with LM Studio's chat completions API
- Handles text generation for answering questions
- Uses the `/v1/chat/completions` endpoint

#### 2. `LMStudioEmbedding` (Embeddings)
- Extends `BaseEmbedding` to generate vector embeddings
- Converts text into numerical vectors for similarity search
- Uses the `/v1/embeddings` endpoint

### Vector Index with Persistence

The application uses a persistent vector index stored in `./wcag_index/`:
- **First run**: Loads the PDF, creates embeddings, builds the index, and saves to disk (~2-5 minutes)
- **Subsequent runs**: Loads the pre-built index from disk (instant startup)

This significantly improves performance on repeated runs.

### RAG Query Pipeline

1. **User asks a question** (e.g., "What does WCAG require for dragging movements?")
2. **Question is embedded** using the embedding model
3. **Vector similarity search** retrieves the top 5 most relevant chunks from the PDF
4. **Context + question** are sent to the LLM
5. **LLM generates an answer** grounded in the retrieved context
6. **Response includes source citations** with relevance scores

## Usage

### Running the Main RAG Application

```bash
python rag_wcag.py
```

**Expected Output:**
```
Loading existing vector index from disk...

QUESTION:
What does WCAG 2.2 require when a user interface relies on dragging movements?

ANSWER:
According to WCAG 2.2, all functionality that uses a dragging movement for operation 
can be achieved by a single pointer without dragging, unless dragging is essential...

--- Retrieved source snippets ---

[1] score=0.769
How to Meet Dragging Movements...
```

### Testing LM Studio Connection

```bash
python test_lmstudio_llm.py
```

This simple test verifies that LM Studio is running and responding correctly.

### Modifying the Query

Edit the `question` variable in `rag_wcag.py` (around line 134):

```python
question = "Your custom question about WCAG here"
```

### Adjusting Retrieval Parameters

In `rag_wcag.py`, modify the query engine configuration:

```python
query_engine = index.as_query_engine(
    similarity_top_k=5,  # Number of relevant chunks to retrieve (default: 5)
)
```

### Changing Models

Update the model names in `rag_wcag.py`:

```python
llm = LMStudioLLM(
    model="your-llm-model-name",  # Change this
    api_base=API_BASE,
)

embed_model = LMStudioEmbedding(
    model="your-embedding-model-name",  # Change this
    api_base=API_BASE,
)
```

## Troubleshooting

### "Connection refused" or "Failed to connect"
- Ensure LM Studio is running and the local server is started
- Verify the server is on `http://localhost:1234`
- Check LM Studio's server logs for errors

### "Model not found" errors
- Confirm the model names in the code match those loaded in LM Studio
- Use the exact model ID shown in LM Studio (e.g., `meta-llama-3.1-8b-instruct`)

### Slow first run
- Initial indexing takes time (2-5 minutes depending on PDF size)
- Subsequent runs are much faster due to persistence
- Consider the size of the PDF and available RAM

### Out of memory errors
- LM Studio models require significant RAM (8GB+ recommended)
- Try using smaller models
- Close other applications

### Index corruption
- Delete the `wcag_index/` folder to rebuild from scratch:
```bash
rm -rf wcag_index/
python rag_wcag.py
```

## Customization

### Use a Different PDF
Replace `wcag-2.2.pdf` with your document and update the filename in `rag_wcag.py`:

```python
documents = SimpleDirectoryReader(
    input_files=["your-document.pdf"]
).load_data()
```

### Add Multiple Documents
Load multiple PDFs or an entire directory:

```python
documents = SimpleDirectoryReader(
    input_dir="./documents"
).load_data()
```

### Adjust LLM Temperature
Control response randomness (0.0 = deterministic, 1.0 = creative):

```python
def complete(self, prompt: str, **kwargs) -> CompletionResponse:
    resp = requests.post(
        f"{self.api_base}/chat/completions",
        json={
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,  # Adjust this value
        },
    )
```

## Benefits of This Approach

✅ **Fully Local** - No data sent to external APIs  
✅ **Privacy-First** - Sensitive documents stay on your machine  
✅ **Cost-Free** - No per-token API charges  
✅ **Customizable** - Full control over models and parameters  
✅ **Fast Retrieval** - Vector search with persistent indexing  
✅ **Source Citations** - Answers include relevance scores and source snippets  

## License

This project is provided as-is for educational and development purposes.

WCAG 2.2 is © W3C (MIT, ERCIM, Keio, Beihang).
