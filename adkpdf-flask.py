from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# app = Flask(__name__)

# Extract text from the uploaded PDF


def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Split text into chunks


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and save the vector store


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational chain with GPT


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context just say, "answer is not available in the context". 
    Don't provide a wrong answer.
    Context: {context}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process the user question


def process_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# API route to handle PDF and question


@app.route("/process_pdf", methods=["POST"])
def process_pdf():
    if "pdf" not in request.files or "question" not in request.form:
        return jsonify({"error": "Please provide both PDF and a question."}), 400

    pdf_file = request.files["pdf"]
    if pdf_file:
        print(f"Received PDF: {pdf_file.filename}")
    else:
        print("PDF file is missing")
    user_question = request.form["question"]

    # Extract text from PDF
    raw_text = get_pdf_text(pdf_file)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)

    # Get AI response
    response = process_user_input(user_question)

    return jsonify({"answer": response})


if __name__ == "__main__":
    app.run(debug=True)
