# app.py
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import os

# Importing internal's
from ChatBot import PdfChatbot
from PdfProcessor import  PdfProcessor


# importing keys
from key import GROQ_API_KEY

app = Flask(__name__)
app.secret_key = 'janina_secret_key'  # Change this to a secure secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# initialize pdf processor
pdfProcessor=PdfProcessor()
# Initialize chatbot
chatbot = PdfChatbot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            chain = pdfProcessor.process_pdf(filepath)
            chatbot.chain=chain
            session['pdf_processed'] = True
            return jsonify({'success': 'PDF processed successfully'})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/ask', methods=['POST'])
def ask_question():
    if not session.get('pdf_processed'):
        return jsonify({'error': 'Please upload a PDF first'})
    
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'})
    
    try:
        answer = chatbot.ask_question(question)
        return jsonify({'answer': answer})
    except Exception as e:
        print("Error Occured In Asking Question :\n")
        print(e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)