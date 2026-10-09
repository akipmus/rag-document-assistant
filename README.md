# RAG Document Assistant

A local Streamlit app for asking questions about PDF documents. It extracts document text, retrieves relevant passages with sentence embeddings and FAISS, then generates answers with a local Ollama model.

## Requirements

- Windows, Python, and Git (optional)
- [Ollama](https://ollama.com/download) with the `llama3.2:3b` model
- Tesseract OCR for scanned or image-only PDFs

Install Ollama, then download the model:

```powershell
ollama pull llama3.2:3b
```

Install Tesseract OCR for Windows if you need to process scanned PDFs. Restart the terminal after installation so the `tesseract` command is available.

## Setup

Open PowerShell in the project directory and create the virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual environment activation, run the app using the environment's Python directly as shown below.

## Run

```powershell
python -m streamlit run app.py
```

Upload a PDF in the app, wait for processing to finish, enter a question, then press Enter or click **Generate answer**. Ollama must be running locally. On Windows, it normally starts with Ollama; check with:

```powershell
ollama list
```

## How it works

- `app.py` provides the Streamlit interface and PDF upload flow.
- `rag_pipeline.py` extracts text, runs OCR when the PDF has no embedded text, splits text into overlapping chunks, retrieves passages, and asks Ollama to answer from the document.
- `requirements.txt` lists the Python dependencies.
- `data/` is available for keeping sample PDFs; upload documents through the app to process them.

The app can only answer from text it extracts or recognizes. OCR may misread low-quality scans, and generated answers can still be wrong, so verify answers against the supporting passages and original PDF.
