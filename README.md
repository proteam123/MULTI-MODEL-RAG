# Multimodal Document Assistant 📄🤖

## 📌 Project Description

A Multimodal RAG application that allows users to interact with PDF documents using natural language. It extracts text, tables, and images from PDFs, processes visual content using vision LLMs, and retrieves relevant information to generate accurate answers.

The application combines text-based and image-based retrieval to provide an interactive document question-answering experience through a Streamlit interface.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Frontend:** Streamlit
- **PDF Processing:** PyMuPDF
- **LLM Provider:** Groq
- **Vision Model:** Groq Vision LLM
- **Embeddings:** Hugging Face Embeddings
- **Vector Database:** Pinecone
- **Local Database:** SQLite
- **Environment Management:** Python-dotenv
- **Development Environment:** VS Code

## ⚙️ Process Pipeline

1. **PDF Upload:** Users upload PDF documents through the Streamlit interface.
2. **Document Extraction:** PyMuPDF extracts text, tables, and embedded images.
3. **Visual Processing:** Groq Vision LLM analyzes and summarizes embedded visual content.
4. **Text Chunking:** Extracted text is divided into smaller chunks for efficient retrieval.
5. **Embedding Generation:** Hugging Face embedding models convert text chunks into vector representations.
6. **Vector Storage:** Generated embeddings are stored in Pinecone.
7. **Query Processing:** Users ask questions through the chat interface.
8. **Similarity Search:** Pinecone retrieves the top 5 relevant chunks.
9. **LLM Processing:** Retrieved context is passed to the text or vision LLM to generate an answer.
10. **Response Generation:** The final answer is displayed in the Streamlit chat interface.

### Additional Features
- Persistent chat history using SQLite.
- Multiple PDF document indexing.
- Re-indexing support for existing documents.
- Image and chart retrieval using `/show image`, `/show chart`, and `/show graph` commands.
- Local PDF storage and shared Pinecone namespace.

## 📚 What I Learned

- Understanding Multimodal RAG architecture and its pipeline.
- Extracting text, tables, and images from PDF documents.
- Working with vision-based LLMs for visual content understanding.
- Generating embeddings using Hugging Face models.
- Implementing vector similarity search using Pinecone.
- Building document retrieval and question-answering systems.
- Combining text retrieval with visual information retrieval.
- Managing persistent chat history using SQLite.
- Developing interactive AI applications using Streamlit.

## 🚀 Future Improvements

- Support additional document formats such as DOCX, PPTX, and scanned PDFs.
- Improve table extraction and structure preservation.
- Implement hybrid search combining keyword and vector retrieval.
- Add reranking models to improve retrieval accuracy.
- Support multiple embedding models.
- Improve image and chart extraction, including vector-based charts.
- Add source citations with PDF page numbers.
- Implement user authentication and document-level access control.
- Optimize retrieval speed and support larger document collections.

## ▶️ How to Run the Project

**1. Clone the repository**
```bash
git clone <repository-url>
cd <project-folder>
```

**2. Create a virtual environment**
```powershell
python -m venv .venv
```

**3. Install dependencies**
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**4. Configure environment variables**

Create or edit the `.env` file and add your Groq and Pinecone API credentials.

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Also configure the model IDs and other settings required by the project.

**5. Run the Streamlit application**
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**6. Access the application**

Open the local Streamlit URL displayed in the terminal.

**7. Upload and index documents**
- Upload PDF files from the Recent Documents section.
- Select a PDF and click **Index Selected PDF**.
- Wait for the document processing and indexing to complete.

**8. Ask questions**

Use the chat interface to ask questions about your indexed documents.

Example:
```text
Summarize this document.

What are the main topics discussed?

Explain the diagram on page 5.

 /show chart What does the chart represent?
```

---

⭐ Built to explore Multimodal AI, Vision LLMs, Vector Databases, and Retrieval-Augmented Generation.
