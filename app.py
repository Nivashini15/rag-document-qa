import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings

# Import all the amazing modules you just built!
from src.document_loader import load_pdf_document
from src.embeddings import process_documents
from src.vector_db import create_and_save_vector_db, load_vector_db
from src.llm_generator import setup_qa_chain

# --- Page Configuration ---
st.set_page_config(page_title="Intelligent Document Q&A", page_icon="📄", layout="centered")
st.title(" Intelligent Document Q&A System")
st.markdown("Upload a PDF document and ask questions about its content directly!")

# --- Sidebar: Document Upload & Processing ---
with st.sidebar:
    st.header(" 1. Upload Document")
    uploaded_file = st.file_uploader("Upload your PDF here", type=["pdf"])
    
    if uploaded_file is not None:
        # Save the file temporarily in the data folder so our loader can read it
        temp_file_path = os.path.join("data", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")
        
        st.header(" 2. Process Document")
        if st.button("Build AI Brain "):
            with st.spinner("Extracting text and building vector database..."):
                docs = load_pdf_document(temp_file_path)
                if docs:
                    chunks, embed_model = process_documents(docs)
                    create_and_save_vector_db(chunks, embed_model)
                    st.success(" Database built! You can now ask questions.")
                else:
                    st.error(" Failed to read the PDF.")

# --- Main Area: Q&A Chat ---
st.header(" 3. Chat with your Document")

# Initialize a chat history to make it feel like a real conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# The chat input box at the bottom of the screen
if user_query := st.chat_input("Ask a question about your PDF..."):
    
    # 1. Display the user's question immediately
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # 2. Generate and display the AI's answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Load the database and connect the LLM
                embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                db = load_vector_db(embed_model)
                
                if db:
                    qa_chain = setup_qa_chain(db)
                    response = qa_chain.invoke({"query": user_query})
                    answer = response['result']
                    
                    # Display the answer and save it to history
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.warning(" Please process a document in the sidebar first!")
                    
            except Exception as e:
                st.error(f" An error occurred: {e}")