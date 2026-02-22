import os
from langchain_community.vectorstores import FAISS

# We will save our database to the 'vectorstore' folder we created earlier
DB_DIR = "../vectorstore"

def create_and_save_vector_db(chunks, embedding_model):
    """
    Takes the text chunks and embedding model, creates a FAISS vector 
    database, and saves it locally so we don't have to re-process the PDF every time.
    """
    print(" Building the FAISS vector database...")
    
    # This automatically embeds the chunks and indexes them
    vector_db = FAISS.from_documents(chunks, embedding_model)
    
    print(f" Saving the database locally to '{DB_DIR}'...")
    vector_db.save_local(DB_DIR)
    
    print(" Vector database successfully built and saved!")
    return vector_db

def load_vector_db(embedding_model):
    """
    Loads the previously saved FAISS database from the local disk.
    """
    if not os.path.exists(DB_DIR):
        print(f" Error: No saved database found at {DB_DIR}.")
        return None
        
    print(" Loading the existing FAISS database...")
    # Note: allow_dangerous_deserialization is a required security flag in LangChain 
    # when loading local pickle files!
    vector_db = FAISS.load_local(
        DB_DIR, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    print(" Database loaded and ready for queries!")
    return vector_db

# --- Quick Test Block ---
if __name__ == "__main__":
    # Import the functions we just built in Modules 1 and 2!
    from document_loader import load_pdf_document
    from embeddings import process_documents
    
    sample_pdf = "../data/1706.03762v7.pdf" # Replace with your actual PDF filename
    
    print("--- Starting Full Ingestion Pipeline Test ---")
    docs = load_pdf_document(sample_pdf)
    
    if docs:
        chunks, embed_model = process_documents(docs)
        
        # 1. Create and save the DB
        db = create_and_save_vector_db(chunks, embed_model)
        
        # 2. Test a retrieval! 
        query = "What is the main architecture proposed in this paper?"
        print(f"\n Testing similarity search for: '{query}'")
        
        # Fetch the top 2 most mathematically relevant chunks
        results = db.similarity_search(query, k=2) 
        
        for i, res in enumerate(results):
            print(f"\n--- Top Result {i+1} ---")
            print(res.page_content[:200] + "...\n")