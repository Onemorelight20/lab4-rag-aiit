import os
from typing import List, Dict, Any, Tuple
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from rag_core import VectorStoreManager
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = os.getenv("LLM_MODEL", model_name)
        self.vector_manager = VectorStoreManager()
        self.llm = ChatOllama(model=self.model_name)
        
        self.system_prompt = "Answer the question based only on the following context:"

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.system_prompt = prompt

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join([d.page_content for d in docs])

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format chat history into a string."""
        formatted_history = ""
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                formatted_history += f"User: {content}\n"
            elif role == "assistant":
                formatted_history += f"Assistant: {content}\n"
        return formatted_history

    def generate_response(self, question: str, history: List[Dict[str, str]] = None, k: int = 3) -> Tuple[str, List[Tuple[Document, float]]]:
        """
        Generate a response to the question using RAG.
        Returns the answer string and a list of (Document, score) tuples.
        """
        docs_and_scores = self.vector_manager.similarity_search_with_score(question, k=k)
        docs = [doc for doc, _ in docs_and_scores]
        
        context_str = self._format_docs(docs)
        
        history_str = self._format_history(history) if history else ""
        
        template = f"""{self.system_prompt}

        Context:
        {{context}}

        Chat History:
        {history_str}

        Question: {{question}}

        Answer:"""
        
        prompt_template = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": lambda x: context_str, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return chain.stream(question), docs_and_scores

    def retrieve_context(self, question: str, k: int = 3) -> List[Tuple[Document, float]]:
        return self.vector_manager.similarity_search_with_score(question, k=k)
