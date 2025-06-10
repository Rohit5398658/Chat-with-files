from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd
import sqlite3
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile

app = Flask(__name__)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

def dataframe_to_text(df):
    text = ""
    for _, row in df.iterrows():
        for col in df.columns:
            text += f"{col}: {row[col]}\n"
        text += "\n"
    return text

def process_pdf(file):
    pdf_reader = PdfReader(file)
    return "".join(page.extract_text() or "" for page in pdf_reader.pages)

def process_csv(file):
    df = pd.read_csv(file)
    return dataframe_to_text(df)

def process_excel(file):
    xls = pd.ExcelFile(file)
    text = ""
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        text += f"Sheet: {sheet_name}\n"
        text += dataframe_to_text(df)
        text += "\n\n"
    return text

def process_sqlite(file):
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(file.read())

    text = ""
    conn = sqlite3.connect(temp_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table_name in tables:
        table_name = table_name[0]
        text += f"Table: {table_name}\n"
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        text += dataframe_to_text(df)
        text += "\n\n"

    conn.close()
    os.remove(temp_path)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, respond with: "Answer is not available in the context."

    Context:\n {context}
    Question:\n {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

@app.route("/process", methods=["POST"])
def process_files():
    files = request.files.getlist("files")
    full_text = ""
    for file in files:
        if file.filename.endswith(".pdf"):
            full_text += process_pdf(file)
        elif file.filename.endswith(".csv"):
            full_text += process_csv(file)
        elif file.filename.endswith(".xlsx"):
            full_text += process_excel(file)
        elif file.filename.endswith(".db"):
            full_text += process_sqlite(file)
    chunks = get_text_chunks(full_text)
    get_vector_store(chunks)
    return jsonify({"status": "success", "message": "Processing complete!"})

@app.route("/query", methods=["POST"])
def answer_query():
    data = request.json
    user_question = data.get("question")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return jsonify({"response": response.get("output_text", "No answer generated.")})

if __name__ == "__main__":
    app.run(debug=True)
