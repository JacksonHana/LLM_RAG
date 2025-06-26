import os
from qdrant_client import QdrantClient
from langchain.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.chatmodels import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

EMBEDDING_MODEL_NAME = os.getenv("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
if not QDRANT_URL:
    raise ValueError("QDRANT_URL is not set in the environment variables.")

# Collect the collection name from the environment variable or use a default
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "phap_luat")

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL)

def create_retrieval_qa_chain(collection_name: str) -> RetrievalQA:
    """
    Create a RetrievalQA chain using Qdrant and Google Generative AI.
    
    Args:
        collection_name (str): The name of the Qdrant collection to use.
        
    Returns:
        RetrievalQA: The configured RetrievalQA chain.
    """
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}  # Adjust device as needed
    )
    if not embeddings:
        raise ValueError("Failed to initialize embeddings. Check the EMBEDDING_MODEL_NAME.")
    

    # Initialize the chat model
    chat_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

    # Create the RetrievalQA chain
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=qdrant_client.as_retriever(
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
    )
    
    return retrieval_qa_chain