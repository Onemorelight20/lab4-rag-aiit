# Lab 4: RAG System with Local Llama

This project implements a Retrieval-Augmented Generation (RAG) system using Streamlit, ChromaDB, and Ollama.

## Prerequisites

1.  **Ollama**: Ensure Ollama is installed and running.
2.  **Model**: Pull the Llama 3.1 model:
    ```bash
    ollama pull llama3.1:8b
    ```

## Installation

1.  Create a virtual environment (if not already active):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the App**:
    ```bash
    streamlit run app.py
    ```
2.  **Upload Documents**: Use the sidebar to upload PDF or Text files (e.g., `Лабораторні роботи 3-4.pdf`).
3.  **Process**: Click "Process Documents" to ingest them into the vector database.
4.  **Chat**: Ask questions in the main chat interface.
5.  **View Context**: Expand "View Retrieved Context & Scores" to see what the model is reading.

## Evaluation

To run the evaluation script:
```bash
python evaluate.py
```
This will run a set of predefined questions against your loaded vector database.
