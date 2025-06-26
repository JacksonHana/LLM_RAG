import os
from rag_chain import create_retrieval_qa_chain, COLLECTION_NAME, EMBEDDING_MODEL_NAME
import gradio as gr

# Set environment variable for the Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Path to PDF
PDF_PATH = "data/Pháp luật đại cương (tái bản lần 1) - ThS Nguyễn Thị Hồng Vân (Chủ biên).pdf"

# Initialize the QA chain using the dynamic function
qa_chain = create_retrieval_qa_chain(pdf_path=PDF_PATH, collection_name="phap_luat", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")


def chatbot(question):
    if not question.strip():
        return "Question: Please enter a valid question."
    
    try:
        result = qa_chain(query=question)
        answer = result['result']
        sources = result.get('source_documents', [])
        
        if sources:
            answer += "\n\nAnswers are based on the following sources:\n"
            for i, doc in enumerate(sources):
                content = doc.page_content.strip().replace("\n", " ")
                answer += f"\n[{i+1}] {content[:200]}..."
        
        return answer
    except Exception as e:
        return f"Error: {e}"
# Create a Gradio interface for the chatbot
gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="Chatbot  about Law",
).launch(server_name="0.0.0.0", server_port=7860)