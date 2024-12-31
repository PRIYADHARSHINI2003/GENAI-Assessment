# GENAI-Assessment

# RAG-based PDF QA System

This project provides a system for answering questions based on the content of uploaded PDF files. It utilizes Flask for the backend, FAISS for vector storage, and LangChain for processing and querying the documents. Users can upload PDF files, which are indexed for efficient retrieval, and then ask questions to receive answers derived from the uploaded content. The system uses the Ollama models (OllamaEmbeddings and ChatOllama) for embedding generation and answer generation, ensuring high-quality and relevant responses.

![image](https://github.com/user-attachments/assets/4e77112e-3bed-4dae-9782-a88589e6ef86)

![image](https://github.com/user-attachments/assets/0a9e06ec-e16e-46e4-8dc5-9b31e498e6bb)

---

## Project Structure

### **Files and Directories**

- **app**: Main Python source file containing the Flask application.
- **Evaluation_metrics**: A Jupyter Notebook for evaluating the system's performance.
- **data/**: Directory to store any data files.
- **myenv1/**: Virtual environment for managing dependencies.
- **templates/**: Folder containing HTML templates (e.g., `index.html`) for the Flask app.
- **.env**: Environment file for managing sensitive configurations like API keys and credentials.

---

## Features

### 1. **PDF Upload and Processing**

- Users can upload multiple PDF files.
- Extracts content from PDFs using `PyMuPDFLoader`.
- Splits text into smaller chunks using `RecursiveCharacterTextSplitter` for better embedding.

### 2. **FAISS Indexing**

- Embeds document chunks using `OllamaEmbeddings`.
- Stores embeddings in a FAISS index for efficient retrieval.

### 3. **Question Answering**

- Retrieves relevant context using FAISS.
- Uses `ChatOllama` with a predefined prompt template for generating accurate answers.
- Provides bullet-point answers based on the retrieved context.

---
### **Flask Application**
![image](https://github.com/user-attachments/assets/fe146837-3537-4d0a-b79e-7bce5f243f9d)

- Flask application rendered in localhost
- Use your own PDF or use sample pdf from the data folder in the project directory


---

## Setup Instructions

### **1. Clone the Repository**

```bash
$ git clone <repository-url>
$ cd <repository-folder>
```

### **2. Set Up Virtual Environment**

```bash
$ python -m venv myenv1
$ source myenv1/bin/activate  # On Windows, use myenv1\Scripts\activate
```

### **3. Install Dependencies**

```bash
$ pip install -r requirements.txt
```
### **4. Download and Set Up Ollama**

- Download and install the Ollama CLI tool from the [Ollama website](https://ollama.ai/).
- Pull the required models for this project by running:
  ```bash
  $ ollama pull nomic-embed-text
  $ ollama pull phi3:mini
  ```

### **5. Add Environment Variables** (For langsmith tracing purpose only)

- Create a `.env` file in the root directory with the following variables:

```
LANGCHAIN_API_KEY = your_api_key
```

### **6. Run the Application**

```bash
$ python app.py
```

---

## Dependencies

- Python 3.x
- Flask
- LangChain
- FAISS
- PyMuPDF
- Dotenv

---

## Author

**Priyadharshini A R**
