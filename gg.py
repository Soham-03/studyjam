


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables and configure API
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API key is missing. Please check your .env file.")
genai.configure(api_key=api_key)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None returned by extract_text
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to initialize and return the conversational chain
def get_conversational_chain():
    prompt_template = """
    Jimmy, the AI study planner, helps students by developing personalized and detailed study plans based on their syllabus, learning styles, and upcoming exam schedules. When a student uploads their syllabus and other relevant notes in PDF form, Jimmy processes this information and tailors the study plan to fit the student’s specific needs. The plan includes:

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
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user interaction and provide responses
def user_input(question):
    # question += "Generate a detailed study plan based on the syllabus..."
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Main function for Streamlit application
def main():
    st.set_page_config("Chat with Jimmy, the Study Planner")
    st.header("Chat with Jimmy 📘")

    user_question = st.text_input("How may i help you?")

    if st.button("Ask question"):
        if user_question:
            user_input(user_question)
        else:
            st.error("Please enter a question.")

    with st.sidebar:
        st.title("Upload your syllabus:")
        pdf_docs = st.file_uploader("Choose PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Extracting data from PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed successfully!")

if __name__ == "__main__":
    main()
