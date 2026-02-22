from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf_document(file_path):
    """
    Reads a PDF file from the given path and extracts its text 
    using LangChain's PyPDFLoader.
    """
    if not os.path.exists(file_path):
        print(f" Error: The file {file_path} was not found.")
        return None

    try:
        print(f" Loading document from: {file_path}...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f" Successfully extracted {len(documents)} pages!")
        return documents
        
    except Exception as e:
        print(f" An error occurred while loading the PDF: {e}")
        return None

# --- Quick Test Block ---
# If you run this file directly, it will test the function!
if __name__ == "__main__":
    # Pointing to the 'data' folder we created
    sample_pdf_path = "../data/1706.03762v7.pdf" # Replace with your downloaded PDF name
    
    docs = load_pdf_document(sample_pdf_path)
    
    if docs:
        print("\n Preview of Page 1:")
        # Printing the first 300 characters of the first page
        print(docs[0].page_content[:300] + "...\n")