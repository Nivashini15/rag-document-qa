from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def process_documents(documents):
    """
    Splits the extracted PDF text into smaller chunks and loads 
    the Sentence-BERT model to convert those chunks into vector embeddings.
    """
    # --- Step 1: Chunking the Text ---
    print(" Splitting the document into smaller chunks...")
    
    # We use RecursiveCharacterTextSplitter to keep paragraphs and sentences together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Number of characters per chunk
        chunk_overlap=200, # Overlap prevents cutting a sentence in half
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f" Successfully split the document into {len(chunks)} chunks!")

    # --- Step 2: Setting up Sentence-BERT ---
    print(" Loading Sentence-BERT embedding model (this might take a few seconds)...")
    
    # 'all-MiniLM-L6-v2' is a lightweight, highly effective Sentence-BERT model
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(" Embedding model loaded and ready!")
    
    return chunks, embeddings_model

# --- Quick Test Block ---
if __name__ == "__main__":
    # We can simulate a dummy document just to test if the model downloads and loads correctly
    from langchain_core.documents import Document
    dummy_docs = [Document(page_content="This is a test document to check if the chunking and embedding logic works perfectly. " * 50)]
    
    test_chunks, test_model = process_documents(dummy_docs)
    print(f"\n First chunk preview: {test_chunks[0].page_content[:100]}...")