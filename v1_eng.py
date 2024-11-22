import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os

class PDFChatbot:
    def __init__(self, api_key):
        """
        Initialize the chatbot with your OpenAI API key
        """
        os.environ["OPENAI_API_KEY"] = api_key
        self.chat_history = []
        self.chain = None
        
    def load_pdf(self, pdf_path):
        """
        Load and extract text from a PDF file
        """
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def process_pdf(self, pdf_path):
        """
        Process PDF content and create a conversational chain
        """
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
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts, embeddings)
        
        # Create the conversational chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        )
        
    def ask_question(self, question):
        """
        Ask a question about the PDF content
        """
        if not self.chain:
            return "Please load a PDF first using process_pdf()"
        
        response = self.chain({"question": question, "chat_history": self.chat_history})
        self.chat_history.append((question, response['answer']))
        
        return response['answer']

# Example usage
def main():
    # Initialize chatbot with your OpenAI API key
    chatbot = PDFChatbot("your-api-key-here")
    
    # Process a PDF file
    chatbot.process_pdf("path_to_your_pdf.pdf")
    
    # Ask questions
    while True:
        question = input("\nAsk a question about the PDF (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        answer = chatbot.ask_question(question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()