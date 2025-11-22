# Lab 4: RAG System with Local Llama

This project implements a robust **Retrieval-Augmented Generation (RAG)** system using **Streamlit**, **ChromaDB**, and **Ollama**. It allows users to chat with their documents (PDF, TXT) using a local LLM, providing accurate answers based on the retrieved context.

## Key Features

*   **Local RAG Pipeline**: Uses `llama3.1:8b` (via Ollama) and `ChromaDB` for a completely local and private experience.
*   **Document Processing**: Supports uploading PDF and Text files, or loading from a local directory.
*   **Smart Retrieval**: Uses `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) for semantic search with cosine similarity.
*   **Context-Aware Chat**: Maintains chat history so you can ask follow-up questions.
*   **Transparency**: Displays the retrieved context chunks, their similarity scores, and source filenames.
*   **Advanced Evaluation**: Includes an `evaluate.py` script that uses **LLM-as-a-judge** to score answers on Accuracy, Completeness, and Consistency. It also tests different retrieval parameters (`k`) and saves results to CSV.

## Project Structure

*   `app.py`: The Streamlit frontend application.
*   `rag_engine.py`: Core RAG logic (LLM interaction, prompt management, history).
*   `rag_core.py`: Document processing (loading, chunking) and Vector Database management.
*   `evaluate.py`: Automated evaluation script with parameter variation and ground truth comparison.
*   `lab_report.md`: Detailed report of the laboratory work and experimental results.

## Prerequisites

1.  **Ollama**: Ensure [Ollama](https://ollama.com/) is installed and running.
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
    - `EVAL_MODEL`: Model used for evaluation (default: same as `LLM_MODEL`)
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
4.  **View Context**: Expand "View Retrieved Context & Scores" to see what the model is reading.

## Evaluation

To run the automated evaluation script:
```bash
python evaluate.py
```
This script will:
1.  Run a set of test questions with ground truth answers.
2.  Test with different values of `k` (1, 3, 5).
3.  Use the LLM to score the answers (1-5) on Accuracy, Completeness, and Consistency.
4.  Print a summary and save detailed results to `evaluation_results.csv`.
