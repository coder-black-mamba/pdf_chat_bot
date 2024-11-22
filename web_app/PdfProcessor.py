import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
import os
from langchain_huggingface import HuggingFaceEmbeddings

# importing keys
from key import GROQ_API_KEY

# Initialize OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY



class PdfProcessor:
    def __init__(self):
        self.chain = None
        
    
    def load_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def process_pdf(self, pdf_path):
        # Extract text from PDF
        raw_text = self.load_pdf(pdf_path)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_text(raw_text)
        
        # Create embeddings and vector store
        # embeddings = OpenAIEmbeddings()
        # vectorstore = FAISS.from_texts(texts, embeddings)
        embeddings_model = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_texts(texts, embeddings_model)
        # Create the conversational chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=ChatGroq(model="llama3-8b-8192",api_key=GROQ_API_KEY),
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        )

        return self.chain
        
if __name__ =="__main__":
    print(os.environ);