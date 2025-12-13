import os, glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# The path to the legal_docs PDF files
PDF_SOURCE_DIR = "legal_docs"
DB_FAISS_PATH = "bns_vector_store" # The directory to save the vector pipstore

def create_vector_db():
    """
    Loads the PDF, splits it into chunks, creates embeddings,
    and saves them to a local FAISS vector store.
    """
    pdf_files = glob.glob(os.path.join(PDF_SOURCE_DIR, "*.pdf"))
    
    # Check if PDF file exists
    if not pdf_files:
        print(f"Error: No PDF files found in '{PDF_SOURCE_DIR}'.")
        print("Please add your BNS, IPC, CrPC, and judgment PDFs to that folder.")
        return

    # 1. Load the PDF document
    all_documents = []
    for pdf_path in pdf_files:
        print(f"Loading PDF from: {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
        print(f"Loaded {len(documents)} pages.")
    print(f"Loaded {len(all_documents)} pages from the PDF.")

    # 2. Split the document into chunks
    # This splitter is good at breaking down code and structured text.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(all_documents)
    print(f"Split the document into {len(docs)} chunks.")

    # 3. Create embeddings
    try:
        # Ye mdoular hai thoda sa, yahan pe embeddings ka model badalna toh agent wale mein bhi badalna!
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        print("Hugging Face Embeddings model loaded.")
    except Exception as e:
        print(f"Error initializing Hugging Face Embeddings: {e}")
        return

    # 4. Create a FAISS vector store from the documents and embeddings
    print("Creating vector store... This may take a few minutes.")
    db = FAISS.from_documents(docs, embeddings)
    
    # 5. Save the vector store locally
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store created and saved successfully at: {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()

