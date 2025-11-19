import argparse
import os
from rag_core import VectorStoreManager
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

def evaluate_system():
    # Define test questions and expected answers (ground truth is hard without manual labeling, 
    # so we will focus on generating answers for manual review or LLM-as-a-judge)
    test_questions = [
        "What is the main goal of Lab 4?",
        "What are the tasks involved in the lab?",
        "Which library is recommended for vector embeddings?",
        "How should the text be chunked?",
        "What metrics should be used for evaluation?",
        "What is the structure of the report?",
        "What is RAG?",
        "How many control questions are needed?",
        "What is the role of the vector database?",
        "Which LLM is used in this implementation?"
    ]

    print("Initializing Evaluation...")
    vector_manager = VectorStoreManager()
    retriever = vector_manager.get_vector_store().as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=LLM_MODEL)

    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt_template = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print(f"\nRunning evaluation on {len(test_questions)} questions...\n")
    
    results = []
    
    for i, question in enumerate(test_questions):
        print(f"Q{i+1}: {question}")
        try:
            # Retrieve context first to check relevance
            docs = retriever.invoke(question)
            context_str = format_docs(docs)
            
            # Generate answer
            answer = rag_chain.invoke(question)
            
            print(f"A: {answer}\n")
            print("-" * 50)
            
            results.append({
                "question": question,
                "answer": answer,
                "context_length": len(context_str),
                "num_source_docs": len(docs)
            })
        except Exception as e:
            print(f"Error processing Q{i+1}: {e}")

    # Simple automated metrics (placeholder for more complex evaluation)
    print("\nEvaluation Summary:")
    print(f"Total Questions: {len(results)}")
    print("Please review the answers above for Accuracy, Completeness, and Consistency.")

if __name__ == "__main__":
    evaluate_system()
