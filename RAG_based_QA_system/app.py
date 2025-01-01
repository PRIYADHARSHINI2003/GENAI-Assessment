import math
import os
from flask import Flask, request, render_template, jsonify
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)

retriever = None
rag_chain = None
vector_store = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    global retriever, rag_chain, vector_store

    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400

    files = request.files.getlist('files')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No selected files"}), 400

    # Temporary storage for all chunks
    all_chunks = []

    for file in files:
        # Save each uploaded file
        file_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(file_path)

        # Process the PDF
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)

        # Clean up uploaded file
        os.remove(file_path)

    # Generate embeddings and create FAISS index if not already created
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    if vector_store is None:
        single_vector = embeddings.embed_query("this is some text data")  # Get embedding size
        index = faiss.IndexFlatL2(len(single_vector))
        vector_store = FAISS(embedding_function=embeddings, index=index, 
                             docstore=InMemoryDocstore(), index_to_docstore_id={})

    # Add all chunks to the FAISS index
    vector_store.add_documents(all_chunks)

    # Set up the retriever
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 100, "lambda_mult": 1})


    # Create the RAG chain
    model = ChatOllama(model="phi3:mini")
    prompt = ChatPromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
        Question: {question}
        Context: {context}
        Answer:
        """
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return jsonify({"message": "PDFs processed and indexed successfully!"})


@app.route('/ask', methods=['POST'])
def ask_question():
    global rag_chain

    if not rag_chain:
        return jsonify({"error": "No PDFs processed yet. Upload PDFs first."}), 400

    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    retrieved_docs = vector_store.similarity_search(question)
    print(retrieved_docs)
    # retrieved_context = [
    #     {"page": doc.metadata.get("page", "Unknown"), "content": doc.page_content}
    #     for doc in retrieved_docs
    # ]
    
    retrieved_context = [[doc.metadata.get("page") , doc.metadata.get("source"), doc.page_content] for doc in retrieved_docs]
    print(retrieved_context)

    output = rag_chain.invoke(question)


    # Run the RAG chain with the question

    return jsonify({
        "retrieved_context": retrieved_context,  # List of page contents
        "response": output
    })

if __name__ == '__main__':
    app.run(debug=True)

