import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import hashlib

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def process_file(self, file_path: str) -> List[Document]:
        """Load and chunk a file (PDF or Text)."""
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def process_directory(self, directory_path: str) -> List[Document]:
        """Load and chunk all supported files in a directory."""
        documents = []
        # Load PDFs
        pdf_loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents.extend(pdf_loader.load())
        # Load Text files
        txt_loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
        documents.extend(txt_loader.load())
        
        chunks = self.text_splitter.split_documents(documents)
        return chunks

class VectorStoreManager:
    def __init__(self):
        self.persist_directory = CHROMA_PATH
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None

    def get_vector_store(self):
        """Initialize or get the vector store."""
        if self.vector_store is None:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
        return self.vector_store

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        store = self.get_vector_store()
        # Generate unique IDs based on content hash to prevent duplicates
        ids = [hashlib.md5(doc.page_content.encode('utf-8')).hexdigest() for doc in documents]
        store.add_documents(documents, ids=ids)
    
    def similarity_search_with_score(self, query: str, k: int = 3):
        """Search for similar documents with scores."""
        store = self.get_vector_store()
        return store.similarity_search_with_score(query, k=k)

    def clear(self):
        """Clear the vector store."""
        if os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
        self.vector_store = None
