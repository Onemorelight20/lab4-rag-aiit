# Lab 4: RAG System with Local Llama

This project implements a Retrieval-Augmented Generation (RAG) system using Streamlit, ChromaDB, and Ollama.

## Prerequisites

1.  **Ollama**: Ensure Ollama is installed and running.
2.  **Model**: Pull the Llama 3.1 model (or your configured model):
    ```bash
    ollama pull llama3.1:8b
    ```

## Installation

1.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configuration**:
    Copy `.env.example` to `.env` and adjust settings if needed:
    ```bash
    cp .env.example .env
    ```
    
    **Available Settings**:
    - `LLM_MODEL`: Model name (default: `llama3.1:8b`)
    - `DOCS_DIR`: Directory to load documents from (default: `./docs`)
    - `CHROMA_PATH`: Path to vector database (default: `./chroma_db`)
    - `EMBEDDING_MODEL`: HuggingFace embedding model (default: `all-MiniLM-L6-v2`)

## Usage

1.  **Run the App**:
    ```bash
    streamlit run app.py
    ```
2.  **Ingest Documents**:
    *   **Option A**: Upload PDF/Text files via the sidebar.
    *   **Option B**: Place files in the `./docs` folder and click "Load from ./docs".
3.  **Chat**: Ask questions in the main chat interface.
4.  **View Context**: Expand "View Retrieved Context & Scores" to see what the model is reading and the similarity scores.

## Evaluation

To run the evaluation script (uses settings from `.env`):
```bash
python evaluate.py
```
