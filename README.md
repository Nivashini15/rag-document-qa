# Intelligent Document Q&A System (RAG Pipeline)

An end-to-end Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions to extract precise, context-aware information.

## Overview
This project implements a robust NLP pipeline to process unstructured document data. It leverages **Sentence-BERT** for high-quality semantic embeddings, a local vector database for rapid similarity search, and a Large Language Model (LLM) to generate accurate answers. The entire system is wrapped in an interactive **Streamlit** web interface, making it easy to deploy and use.

## Key Features
* **Automated Document Ingestion:** Extracts and cleans text from dense PDFs (Module 1).
* **Semantic Chunking & Embedding:** Splits text intelligently and embeds it using Sentence-BERT for accurate retrieval (Module 2).
* **Vector Database Integration:** Efficiently stores and searches document embeddings using FAISS/ChromaDB (Module 3).
* **LLM Generation:** Synthesizes retrieved context to answer user queries naturally without hallucinations (Module 4).
* **Interactive UI:** A sleek, real-time Streamlit application for seamless user interaction (Module 5).

## Tech Stack
* **Language:** Python
* **Frameworks:** LangChain, Streamlit
* **NLP & Embeddings:** Sentence-BERT, Hugging Face
* **Vector Store:** FAISS / ChromaDB

## Project Structure
```text
rag-document-qa/
├── data/                  # Sample PDFs and text documents
├── vectorstore/           # Local vector database storage
├── src/                   # Core application logic
│   ├── document_loader.py # PDF extraction
│   ├── embedding.py       # Chunking & Sentence-BERT logic
│   ├── vector_db.py       # Database creation and retrieval
│   └── llm_generator.py   # LLM answering logic
├── app.py                 # Streamlit UI front-end
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation