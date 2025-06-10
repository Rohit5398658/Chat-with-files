import streamlit as st
import requests

API_URL = "http://localhost:5000"

def main():
    st.set_page_config(page_title="Chat with Files & Databases")
    st.header("Chat with PDF, CSV, Excel, and Databases using Gemini💁")

    user_question = st.text_input("Ask a Question from the Uploaded Files/Database")

    if user_question and st.button("Ask"):
        with st.spinner("Fetching answer..."):
            res = requests.post(f"{API_URL}/query", json={"question": user_question})
            st.write("Reply:", res.json().get("response", "No answer returned."))

    with st.sidebar:
        st.title("Menu:")
        uploaded_files = st.file_uploader(
            "Upload Files (PDF, CSV, Excel, SQLite DB) and click Submit",
            type=["pdf", "csv", "xlsx", "db"],
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if uploaded_files:
                files = [("files", (f.name, f, f.type)) for f in uploaded_files]
                with st.spinner("Processing..."):
                    res = requests.post(f"{API_URL}/process", files=files)
                    st.success(res.json().get("message", "Processed!"))
            else:
                st.warning("No files uploaded.")

if __name__ == "__main__":
    main()
