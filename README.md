# Project: Personalised RAG 

This project implements a Document Query and Retrieval-Augmented Generation (RAG) System that accepts user-uploaded documents and a query. It retrieves relevant information from the documents and generates responses using a Large Language Model (LLM). The pipeline includes semantic chunking, embeddings, and retrieval-based query answering.

## RAG Architecture
![RAG Architecture](/RAG_ARCH.png)

### Install requirements.txt 
```commandline
pip install -m requirements.txt
```
### Download and Install ollama
in [`OllamaDownload`](https://ollama.com/)

### PULL the model in the terminal, 
It will download the model in your local
```commandline
ollama pull llama3.2:1b
```

### Below are the libraries used in this project:

- Flask: For building the API endpoints.
- LangChain: For building LLM-powered pipelines.
- pdf2image: For converting PDF pages to images.
- pytesseract: For extracting text from images via OCR.
- FAISS: For vector store and similarity search.
- HuggingFace Embeddings: For creating text embeddings.
- SentenceTransformer: For embedding generation using pre-trained models.



##  Architecture Overview
Main Components
- Flask API (app.py):
Handles file uploads.
Receives user queries and invokes the RAG pipeline.

- Engine (engine.py):
Processes uploaded documents (semantic chunking, embeddings, retrieval).
Implements Retrieval-Augmented Generation (RAG).

- Utilities (utils.py):
Handles file-specific operations (e.g., text extraction using OCR).


## Step-by-Step Workflow
1. File Upload
Users upload a document and submit a query to the /upload endpoint via a POST request.
Allowed file types include:
PDFs
Images (JPEG, PNG)
Word documents (DOC, DOCX)
Files are stored in the Document directory.
2. Text Extraction (utils.py)
For PDFs: Pages are converted to images using pdf2image.
OCR (Optical Character Recognition) is performed on the images using pytesseract.
Extracted text is cleaned to remove excess whitespaces or newline characters.
3. Semantic Chunking (engine.py)
Text from the document is segmented into semantically meaningful chunks using SemanticChunker.
Chunking ensures that the LLM processes relevant portions of the document.
4. Embedding Generation and Retrieval (engine.py)
Chunks are embedded using a pre-trained embedding model (all-mpnet-base-v2).
FAISS is used to create a vector store for efficient similarity search.
A user query is embedded and matched with relevant chunks.
5. Query Answering (engine.py)
The OllamaLLM model generates responses based on the retrieved chunks.
RetrievalQA chain combines retrieval and LLM-based answering.

### Endpoints
1. Upload and Query
- Endpoint: /upload
- Method: POST
- Parameters:

Query: A user-provided question (query) about the uploaded document.
File: A document file (PDF, JPEG, DOCX, etc.).

- Response:

200: Successful response with the generated answer.

400: Client-side errors (e.g., missing query, invalid file type).

500: Internal server errors during processing.

### Directory Structure
```commandline

├── app.py              # Flask API
├── engine.py           # RAG pipeline implementation
├── utils.py            # Utilities for text extraction
├── requirements.txt    # Python dependencies
├── Document/           # Directory to store uploaded files
└── README.md           # Project documentation

```

### Testing
Example Test File
Place a test document in the Document folder.

Use the following curl command:
```commandline

curl -X POST http://127.0.0.1:5000/upload -F "query=Test" -F "file=@/path/to/your/file.pdf"

```