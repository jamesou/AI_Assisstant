import PyPDF2
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdfs(pdf_paths):
    """
    Load and extract text from a list of PDF files.
    
    Args:
        pdf_paths (list): List of paths to PDF files.
        
    Returns:
        list: List of document texts.
    """
    documents = []
    for path in pdf_paths:
        try:
            with open(path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                documents.append(text)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    return documents


def load_urls(urls):
    """
    Scrape and extract text from a list of URLs.
    
    Args:
        urls (list): List of URLs to scrape.
        
    Returns:
        list: List of document texts.
    """
    documents = []
    for url in urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for script_or_style in soup(['script', 'style']):
                    script_or_style.decompose()
                text = ' '.join(soup.stripped_strings)
                documents.append(text)
            else:
                print(f"Failed to retrieve {url}")
        except Exception as e:
            print(f"Error loading {url}: {e}")
    return documents

def preprocess_documents(doc_texts, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks.
    
    Args:
        doc_texts (list): List of document texts.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between chunks.
        
    Returns:
        list: List of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for text in doc_texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks




def create_vector_store(documents, embedding_model, collection_name="my_collection"):
    """
    Create and store embeddings in Qdrant vector store.
    
    Args:
        documents (list): List of document texts.
        embedding_model: Hugging Face embedding model.
        collection_name (str): Qdrant collection name.
        
    Returns:
        Qdrant: Initialized Qdrant vector store.
    """

    # Convert texts to LangChain Document objects
    docs = [Document(page_content=doc) for doc in documents]
    print(f"docs:{docs}")
    # Create the Qdrant vector store from documents
    vectorstore = Qdrant.from_documents(
        documents=docs,
        embedding=embedding_model,
        url="http://localhost:6333",
        collection_name=collection_name
    )
    return vectorstore

def main():
    # Define document sources (PDFs, URLs)
    pdf_paths = ['pdfs/Nestech Onboarding Manual July 2024_signed.pdf']
    urls = ['https://nestech.co.nz/', 'https://blog.langchain.dev/announcing-langchain-v0-3/']
    
    # Step 1: Load the documents
    pdf_docs = load_pdfs(pdf_paths)
    url_docs = load_urls(urls)
    
    # Combine all documents
    all_docs = pdf_docs + url_docs
    print(f"Total documents loaded: {len(all_docs)}")
    
    # Step 2: Preprocess and chunk the documents
    chunks = preprocess_documents(all_docs)
    print(f"Total document chunks: {len(chunks)}")
    
     # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 3: Create and populate the Qdrant vector store
    vectorstore = create_vector_store(chunks, embedding_model)
    print(f"Vector store created and populated.vectorstore:{vectorstore}")
     

if __name__ == "__main__":
    main()
