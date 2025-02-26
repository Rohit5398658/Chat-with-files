# RAG_WITH_GEMINI

# 📄 Chat with PDFs using Google Gemini AI 💁‍♂️

This project is a **Conversational AI system** powered by **Retrieval-Augmented Generation (RAG)**. It allows users to upload PDF documents and interactively ask questions about the content. The system intelligently retrieves the most relevant information from the documents and provides detailed, context-aware answers using **Google Gemini AI**.

## 🧠 Key Features:
- **Conversational Question Answering**: Upload any PDF and get answers to your questions in natural language.
- **Powered by RAG**: Combines the power of **retrieval** (using FAISS) and **generation** (using Google’s Gemini AI) to give accurate, context-driven answers.
- **Google Generative AI Embeddings**: Efficiently processes large amounts of text and creates embeddings to enable fast and relevant searches.
- **Simple Streamlit Interface**: Intuitive and easy-to-use interface where users can upload multiple PDF files and interact with them.
- **Seamless Document Processing**: Automatically reads, processes, and splits PDF content for efficient question answering.

## 🔍 How It Works:
1. **Upload PDFs**: The user uploads one or more PDF documents through the sidebar.
2. **Text Extraction & Chunking**: The system extracts the text from the PDF files and breaks it into manageable chunks.
3. **Vector Store Creation**: These chunks are embedded using **GoogleGenerativeAIEmbeddings** and stored in a **FAISS vector store** for quick retrieval.
4. **Ask Questions**: The user can ask any question, and the system retrieves relevant information from the PDFs and generates a detailed answer using **Google’s Gemini AI**.
5. **Accurate Answers**: If the information isn't present in the document, the system will inform the user accordingly, ensuring no misleading answers.

## 🛠️ Technology Stack:
- **Streamlit**: For the user interface.
- **PyPDF2**: For PDF text extraction.
- **LangChain**: For managing text chunking and prompt chaining.
- **FAISS**: To efficiently store and search document embeddings.
- **Google Generative AI (Gemini)**: For generating natural language responses.
- **dotenv**: To manage API keys securely.

## 🚀 Getting Started:
### 1. Clone the Repository:
```bash
git clone https://github.com/your-username/chat-with-pdf
cd chat-with-pdf
