# Multimodal Document Assistant — VS Code

This project reorganizes the supplied notebook into a Python application with a Streamlit UI.

## Project flow

PDF upload → PyMuPDF extracts text/tables/images → Groq vision model summarizes embedded visuals
→ Hugging Face embeddings → Pinecone → retrieve top 5 relevant chunks → text LLM or vision LLM
→ answer shown in Streamlit.

## Windows PowerShell setup

```powershell
# Open this project folder in VS Code
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

notepad .env
```

Enter valid Groq and Pinecone API keys in `.env`, save it, and restart Streamlit.
The UI can open without keys, but indexing and chat require both. Keep `.env`
private; it is excluded from Git.

## Run the UI

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the Streamlit URL shown in the terminal. Upload PDFs from the sidebar's
**Recent documents** section, then choose a recent PDF and click **Index selected
PDF**. Uploaded PDFs are retained locally under `data/uploads`; indexed vectors
are added to the shared Pinecone namespace so questions can use all indexed
documents. Re-indexing a PDF updates its existing chunks without deleting other
documents.

Use **New chat** and **Previous chats** in the sidebar to create and switch
between persistent conversations. Session messages are stored in the local
`data/assistant.db` database.

To display a matching chart or image, enter `/show graph <question>`,
`/show chart <question>`, or `/show image <question>`. The app searches relevant
PDF pages and uses the vision model to identify and crop the matching visual.
For older indexed files, re-index them to enable page crops.

Then ask questions in the chat box.

## Notes

- Notebook model IDs and provider access can change. If Groq reports that a model is unavailable, update `TEXT_MODEL` or `VISION_MODEL` in `.env` to a model currently enabled for your Groq account.
- The PDF extraction logic finds embedded images in the PDF. Some charts are drawn as vector shapes rather than embedded image objects and may not be extracted as images by this approach.
- The default Pinecone namespace is `fy2026-demo`; it contains chunks from all PDFs indexed by this app.
- Jupyter/IPykernel are not required to run this Streamlit application.
