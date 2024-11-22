# No Module Import Nedded
class PdfChatbot:
    def __init__(self,chain=None):
        self.chain = chain
        self.chat_history = []
    
        
    def ask_question(self, question):
        if not self.chain:
            return "Please upload a PDF first"
        
        response = self.chain({"question": question, "chat_history": self.chat_history})
        self.chat_history.append((question, response['answer']))
        
        return response['answer']
