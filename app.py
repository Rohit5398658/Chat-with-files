import streamlit as st
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

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY is missing. Please check your .env file.")
    st.stop()

genai.configure(api_key=api_key)

def get_file_text(uploaded_files):
    text = ""
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif file.type == "text/csv":
            df = pd.read_csv(file)
            text += dataframe_to_text(df)
        elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                           "application/vnd.ms-excel"]:
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name)
                text += f"Sheet: {sheet_name}\n"
                text += dataframe_to_text(df)
                text += "\n\n"
    return text

def dataframe_to_text(df):
    text = ""
    for _, row in df.iterrows():
        for col in df.columns:
            text += f"{col}: {row[col]}\n"
        text += "\n"
    return text

def get_db_text(db_file):
    text = ""
    temp_path = f"./temp_{db_file.name}"
    
    with open(temp_path, "wb") as f:
        f.write(db_file.getbuffer())

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
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply:", response.get("output_text", "No response generated."))

def main():
    st.set_page_config(page_title="Chat with Files & Databases")
    st.header("Chat with PDF, CSV, Excel, and Databases using Gemini💁")

    user_question = st.text_input("Ask a Question from the Uploaded Files/Database")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader(
            "Upload your Files (PDF, CSV, Excel, SQLite Database) and Click on the Submit & Process Button",
            type=["pdf", "csv", "xlsx", "db"],
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                for file in uploaded_files:
                    if file.name.endswith(".db"):
                        raw_text += get_db_text(file)
                    else:
                        raw_text += get_file_text([file])
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete!")

if __name__ == "__main__":
    main()
