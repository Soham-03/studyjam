from flask import Flask, request, jsonify
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import tempfile
from flask_cors import CORS
import markdown

app = Flask(__name__)
CORS(app)

# Load environment variables and configure Google Generative AI
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API key is missing. Please check your .env file.")
genai.configure(api_key=api_key)

def download_pdf(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name

def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_conversational_chain():
    prompt_template = """
    - **If any question is aksed beyond the context search from the internet and give approriate response
    Jimmy, the AI study planner, helps students by developing personalized and detailed study plans based on their syllabus, learning styles, and upcoming exam schedules. When a student uploads their syllabus and other relevant notes in PDF form, Jimmy uses the {context} processes this information and tailors the study plan to fit the student’s specific needs. The plan includes:

    - **Overall Study Timeline**: Calculating the total time available until the exams start and suggesting daily study durations.
    - **Subject-Specific Plans**: For each subject in the syllabus, providing a breakdown of topics, estimated time to cover each topic, and prioritizing them based on their importance and weightage in the exams.
    - **Learning Techniques**: Recommending different study techniques that align with the student's learning style, such as summarization, self-testing, or group study sessions.
    - **Break and Review Sessions**: Integrating break periods and review sessions to ensure retention and prevent burnout.
    - **Flashcards**: Automatically generating flashcards for key concepts and review points to be used after every 2-3 modules.
    - **Resource Links**: Offering links to additional resources and explanatory content on complex topics, fetched from the internet or based on the AI’s embedded knowledge.
    - **Exam Preparation Tips**: Providing specific strategies for exam preparation, including how to approach different types of questions and manage time during the exam.

    Context:
    {context}

    Question:
    {question}

    Reply:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    data = request.json
    question = data['question']
    pdf_url = data['pdf_url']

    # Download and extract text from PDF
    pdf_path = download_pdf(pdf_url)
    pdf_text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text(pdf_text)

    # Create vector store from text chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    # Load conversational chain
    chain = get_conversational_chain()

    # Get response
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    # html = markdown.markdown(response)
    html_response = markdown.markdown(response["output_text"])
    return jsonify({"response": html_response})
    # return jsonify({"response": response["output_text"]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)