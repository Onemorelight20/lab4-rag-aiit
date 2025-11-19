import streamlit as st
import os
import tempfile
from rag_core import DocumentProcessor, VectorStoreManager
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
DOCS_DIR = os.getenv("DOCS_DIR", "./docs")

# Page config
st.set_page_config(page_title="Lab 4: RAG System", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False

# Sidebar
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Select Model", [LLM_MODEL], index=0)
    
    system_prompt = st.text_area(
        "System Prompt", 
        value="Answer the question based only on the following context:",
        height=100
    )
    
    st.divider()
    st.header("Document Ingestion")
    
    # Option 1: Upload Files
    uploaded_files = st.file_uploader("Upload PDF or Text files", accept_multiple_files=True, type=['pdf', 'txt'])
    
    if st.button("Process Uploaded Files"):
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                processor = DocumentProcessor()
                vector_manager = VectorStoreManager()
                
                all_chunks = []
                for uploaded_file in uploaded_files:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        chunks = processor.process_file(tmp_path)
                        all_chunks.extend(chunks)
                    finally:
                        os.unlink(tmp_path)
                
                if all_chunks:
                    vector_manager.add_documents(all_chunks)
                    st.session_state.vector_store_ready = True
                    st.success(f"Processed {len(all_chunks)} chunks from {len(uploaded_files)} files.")
        else:
            st.warning("Please upload files first.")

    st.divider()
    
    # Option 2: Load from Directory
    if st.button(f"Load from {DOCS_DIR}"):
        if os.path.exists(DOCS_DIR):
            with st.spinner(f"Processing files from {DOCS_DIR}..."):
                processor = DocumentProcessor()
                vector_manager = VectorStoreManager()
                
                chunks = processor.process_directory(DOCS_DIR)
                
                if chunks:
                    vector_manager.add_documents(chunks)
                    st.session_state.vector_store_ready = True
                    st.success(f"Processed {len(chunks)} chunks from directory.")
                else:
                    st.warning("No supported files found in directory.")
        else:
            st.error(f"Directory {DOCS_DIR} does not exist.")

    if st.button("Clear Database"):
        VectorStoreManager().clear()
        st.session_state.vector_store_ready = False
        st.success("Database cleared.")

# Main Chat Interface
st.title(f"🤖 RAG Chat with {model_name}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat logic
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if not st.session_state.vector_store_ready:
            response = "Please upload and process documents first to enable RAG."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            vector_manager = VectorStoreManager()
            retriever = vector_manager.get_vector_store().as_retriever(search_kwargs={"k": 3})
            
            llm = ChatOllama(model=model_name)
            
            template = system_prompt + """
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
            
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream response
            for chunk in rag_chain.stream(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # Show retrieved context with scores
            with st.expander("View Retrieved Context & Scores"):
                # Get docs with scores
                docs_and_scores = vector_manager.get_vector_store().similarity_search_with_score(prompt, k=3)
                for i, (doc, score) in enumerate(docs_and_scores):
                    st.markdown(f"**Source {i+1} (Score: {score:.4f}):**")
                    st.markdown(doc.page_content)
                    st.divider()
