from langchain_community.document_loaders import PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores.qdrant import Qdrant
from langchain.huggingface import HuggingFaceEmbeddings


path = "data/Pháp luật đại cương (tái bản lần 1) - ThS Nguyễn Thị Hồng Vân (Chủ biên).pdf "
# Load the PDF document
loader = PDFLoader(path)
# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust chunk size as needed
    chunk_overlap=200  # Adjust overlap as needed
)
documents = loader.load_and_split(text_splitter=text_splitter)

#create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": "cpu"}  
)

# Initialize Qdrant vector store
qdrant = Qdrant.from_documents(
    documents=documents,
    embedding=embeddings,                              # Replace with your embedding model
    collection_name="your_collection_name",      # Replace with your collection name
    url="http://localhost:6333",                 # Replace with your Qdrant URL
    prefer_grpc=True,
    distance="cosine"                            # distance metric
)