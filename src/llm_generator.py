import os
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

# Explicitly load the .env file from the main folder
load_dotenv("../.env")

def setup_qa_chain(vector_db):
    print("🧠 Connecting to the Large Language Model...")
    
    # 1. Back to the reliable Mistral model!
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens=512,
        temperature=0.3,
        huggingfacehub_api_token=os.environ.get("HF_TOKEN")
    )
    
    # 2. Keep the Chat wrapper so we satisfy the "conversational" requirement!
    chat_model = ChatHuggingFace(llm=llm)
    
    prompt_template = """Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Do not make up information.

    Context: {context}

    Question: {question}
    Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    print("🔗 Linking the LLM to your FAISS database...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model, # Pass the wrapped chat_model here!
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}), 
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("✅ The LLM Brain is fully connected and ready!")
    return qa_chain

# --- Quick Test Block ---
if __name__ == "__main__":
    from vector_db import load_vector_db
    
    print("--- Starting LLM Connection Test ---")
    
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = load_vector_db(embed_model)
    
    if db:
        qa = setup_qa_chain(db)
        
        test_query = "What is the main architecture proposed in this paper?"
        print(f"\n User: {test_query}")
        
        print(" Generating answer...")
        response = qa.invoke({"query": test_query})
        print(f"\n AI Answer: {response['result']}\n")