import argparse
import os
import json
import csv
from datetime import datetime
from rag_engine import RAGSystem
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


GEN_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
EVAL_MODEL = os.getenv("EVAL_MODEL", GEN_MODEL) 

def evaluate_answer(llm, question, answer, context, expected_answer=None):
    """
    Use LLM to evaluate the answer based on Accuracy, Completeness, and Consistency.
    If expected_answer is provided, it is used as a reference.
    """
    eval_template = """You are an expert evaluator for a RAG system.
    Your task is to evaluate the following Answer to the Question based on the provided Context and an optional Expected Answer.
    
    Question: {question}
    Context: {context}
    Expected Answer: {expected_answer}
    Actual Answer: {answer}
    
    Evaluate the Actual Answer on the following criteria:
    1. Accuracy: Is the information correct according to the context and expected answer? (1-5)
    2. Completeness: Does it cover all relevant parts of the context? (1-5)
    3. Consistency: Is the answer logical and consistent? (1-5)
    
    Provide your output as a JSON object with keys: "accuracy", "completeness", "consistency", "reasoning".
    Do NOT output anything else. JSON only.
    """
    
    prompt = ChatPromptTemplate.from_template(eval_template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result_str = chain.invoke({
            "question": question, 
            "answer": answer, 
            "context": context,
            "expected_answer": expected_answer if expected_answer else "N/A"
        })
        # Clean up potential markdown code blocks
        result_str = result_str.replace("```json", "").replace("```", "").strip()
        return json.loads(result_str)
    except Exception as e:
        print(f"Error evaluating: {e}")
        return {"accuracy": 0, "completeness": 0, "consistency": 0, "reasoning": "Failed to parse"}

def evaluate_system():
    # Test cases with ground truth
    test_cases = [
        {
            "question": "What is the main goal of Lab 4?",
            "expected": "To familiarize with RAG architecture, integrate LLM with external data source, and evaluate the system."
        },
        {
            "question": "What is Retrieval-Augmented Generation (RAG)?",
            "expected": "A technique that combines Large Language Models (LLMs) with external data retrieval to improve answer quality and relevance."
        },
        {
            "question": "What are the steps to implement the RAG pipeline?",
            "expected": "Preprocessing (chunking), Vector Database creation, Retrieval (top-k), and Generation (LLM with context)."
        },
        {
            "question": "Which library is recommended for vector embeddings in the lab?",
            "expected": "SentenceTransformers (e.g., all-MiniLM-L6-v2) or HuggingFaceEmbeddings."
        },
        {
            "question": "How should the text be chunked according to the instructions?",
            "expected": "Into fragments of 500-1000 words with overlap."
        },
        {
            "question": "What metrics should be used to evaluate the quality of answers?",
            "expected": "Accuracy (correctness), Completeness (coverage), and Consistency (logic)."
        },
        {
            "question": "What is the role of the vector database?",
            "expected": "To store vector representations (embeddings) of text chunks for efficient similarity search."
        },
        {
            "question": "Explain the concept of 'chunks' in this context.",
            "expected": "Small, manageable pieces of text split from larger documents to fit within the LLM's context window and improve retrieval precision."
        },
        {
            "question": "What is the purpose of the 'history' in the dialogue?",
            "expected": "To provide context from previous turns in the conversation, allowing for follow-up questions and coherent dialogue."
        },
        {
            "question": "How is the relevance of results ranked?",
            "expected": "Using similarity metrics like cosine similarity between the query embedding and chunk embeddings."
        }
    ]

    print(f"Initializing Evaluation with Generation Model: {GEN_MODEL}")
    print(f"Using Evaluation Model: {EVAL_MODEL}")
    
    rag_system = RAGSystem(model_name=GEN_MODEL)
    eval_llm = ChatOllama(model=EVAL_MODEL, format="json") 

    # Parameter variation
    k_values = [1, 3, 5]
    
    print(f"\nRunning evaluation on {len(test_cases)} questions with k={k_values}...\n")
    
    all_results_data = []
    overall_stats = {}

    for k in k_values:
        print(f"\n=== Evaluating with k={k} ===")
        k_results = []
        
        for i, case in enumerate(test_cases):
            question = case["question"]
            expected = case["expected"]
            
            print(f"Q{i+1}: {question}")
            try:
                # 1. Retrieve Context
                docs_and_scores = rag_system.retrieve_context(question, k=k)
                context_str = "\n\n".join([d.page_content for d, _ in docs_and_scores])
                
                # 2. Generate Answer
                response_stream, _ = rag_system.generate_response(question, k=k) 
                
                answer = ""
                for chunk in response_stream:
                    answer += chunk
                
                print(f"A: {answer[:100]}...") 
                
                # 3. Evaluate
                score = evaluate_answer(eval_llm, question, answer, context_str, expected)
                print(f"Scores: {score}")
                
                # Store result
                result_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "k": k,
                    "question": question,
                    "expected_answer": expected,
                    "generated_answer": answer,
                    "accuracy": score.get("accuracy", 0),
                    "completeness": score.get("completeness", 0),
                    "consistency": score.get("consistency", 0),
                    "reasoning": score.get("reasoning", "")
                }
                k_results.append(result_entry)
                all_results_data.append(result_entry)
                
            except Exception as e:
                print(f"Error processing Q{i+1}: {e}")
        
        # Calculate average for this k
        if k_results:
            avg_acc = sum(r['accuracy'] for r in k_results) / len(k_results)
            avg_comp = sum(r['completeness'] for r in k_results) / len(k_results)
            avg_cons = sum(r['consistency'] for r in k_results) / len(k_results)
            
            overall_stats[k] = {
                "accuracy": avg_acc,
                "completeness": avg_comp,
                "consistency": avg_cons
            }
            print(f"--- Average Scores for k={k}: Acc={avg_acc:.2f}, Comp={avg_comp:.2f}, Cons={avg_cons:.2f} ---")

    csv_filename = "evaluation_results.csv"
    fieldnames = ["timestamp", "k", "question", "expected_answer", "generated_answer", "accuracy", "completeness", "consistency", "reasoning"]
    
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(all_results_data)
        
    print(f"\nResults saved to {csv_filename}")

    print("\n=== Final Summary ===")
    for k, metrics in overall_stats.items():
        print(f"k={k}: {metrics}")

if __name__ == "__main__":
    evaluate_system()
