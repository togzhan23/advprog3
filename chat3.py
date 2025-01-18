import streamlit as st
import logging
from langchain_ollama import OllamaLLM
import chromadb
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import chardet
import fitz  # Для работы с PDF (PyMuPDF)

logging.basicConfig(level=logging.INFO)

chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

class EmbeddingFunction:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        vectors = self.model.encode(input)
        if len(vectors) == 0:
            raise ValueError("Empty embedding generated.")
        return vectors

embedding = EmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")

collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"}
)

def add_document_to_collection(documents, ids, metadatas=None):
    try:
        embeddings = []
        for doc in documents:
            if not doc.strip():
                raise ValueError("Cannot add an empty or whitespace-only document.")
            embedding_vector = embedding(doc)
            logging.info(f"Generated embedding for document '{doc}': {embedding_vector}")
            embeddings.append(embedding_vector[0])

        embeddings = np.array(embeddings)

        logging.info(f"Embeddings shape: {embeddings.shape}")
        
        # Ensure metadata is a non-empty dictionary
        if metadatas is None:
            metadatas = [{"article_id": "Unknown"} for _ in documents]  # Use default metadata if none is provided
        
        # Ensure that each entry in metadatas is a valid dictionary
        for meta in metadatas:
            if not isinstance(meta, dict) or not meta:
                raise ValueError("Each metadata entry must be a non-empty dictionary.")
        
        # Add documents with metadata
        collection.add(documents=documents, embeddings=embeddings.tolist(), ids=ids, metadatas=metadatas)
        logging.info(f"Successfully added {len(documents)} documents to the collection.")
    except Exception as e:
        logging.error(f"Error adding document: {e}")
        raise





def preprocess_txt_constitution(file_path):
    articles = []
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        sections = content.split("Article ")
        for section in sections[1:]:
            header, *body = section.split("\n", 1)
            article_id = header.split(":")[0].strip()
            text = "Article " + header.strip() + "\n" + "\n".join(body).strip()
            articles.append({"id": article_id, "text": text})
    return articles

def preprocess_pdf_constitution(pdf_path):
    articles = []
    try:
        with fitz.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf:
                full_text += page.get_text()

        sections = full_text.split("Article ")
        for section in sections[1:]:
            header, *body = section.split("\n", 1)
            article_id = header.split()[0].strip()
            text = "Article " + header.strip() + "\n" + "\n".join(body).strip()
            articles.append({"id": article_id, "text": text})
    except Exception as e:
        logging.error(f"Error reading or processing PDF: {e}")
        raise
    return articles

def query_documents_from_chromadb(query_text, n_results=1):
    results = collection.query(query_texts=[query_text], n_results=n_results)
    documents = results["documents"]
    metadatas = results.get("metadatas", [])
    return documents, metadatas


def query_with_ollama(prompt, model_name):
    try:
        logging.info(f"Sending prompt to Ollama with model {model_name}: {prompt}")
        llm = OllamaLLM(model=model_name)
        response = llm.invoke(prompt)
        logging.info(f"Ollama response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error with Ollama query: {e}")
        return f"Error with Ollama API: {e}"


# Delete all documents from the collection
def delete_all_documents_from_collection():
    try:
        # Fetch all document IDs
        document_ids = [doc_id for doc_id in collection.get()['ids']]
        
        if not document_ids:
            logging.info("No documents to delete.")
            return
        
        # Delete the documents by their IDs
        collection.delete(ids=document_ids)
        logging.info(f"Deleted {len(document_ids)} documents from the collection.")
        
        # Optionally: Confirm the deletion by checking if any documents are left
        remaining_docs = collection.get()["documents"]
        if remaining_docs:
            logging.info(f"Remaining documents: {len(remaining_docs)}")
        else:
            logging.info("No documents remaining in the collection.")
        
    except Exception as e:
        logging.error(f"Error deleting documents from collection: {e}")


def retrieve_and_answer(query_text, model_name):
    try:
        retrieved_docs, metadatas = query_documents_from_chromadb(query_text)

        context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

        # Ensure metadata is properly accessed
        if metadatas:
            article_id = metadatas[0].get("article_id", "Unknown Article") if isinstance(metadatas[0], dict) else "Unknown Article"
        else:
            article_id = "Unknown Article"

        if article_id == "Unknown Article":
            logging.warning(f"Article ID not found for the query: {query_text}")

        augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
        response = query_with_ollama(augmented_prompt, model_name)

        return response, article_id
    except Exception as e:
        logging.error(f"Error in retrieve_and_answer: {e}")
        return "Sorry, I couldn't find an answer.", "Unknown Article"

documents = ["This is the content of Article 1.", "This is the content of Article 2."]
ids = ["doc1", "doc2"]
metadatas = [{"article_id": "Article 1"}, {"article_id": "Article 2"}]  # Proper metadata for each document

add_document_to_collection(documents, ids, metadatas)

   

st.title("Chat with Ollama")

model = st.sidebar.selectbox("Choose a model", ["llama3.2", "llama3.2"])

if not model:
    st.warning("Please select a model.")

menu = st.sidebar.selectbox("Choose an action", ["Show Documents in ChromaDB", "Clear All Documents in ChromaDB", "Add New Document to ChromaDB as Vector", "Ask Ollama a Question"])

if menu == "Clear All Documents in ChromaDB":
    st.subheader("Clear All Documents in ChromaDB")
    if st.button("Clear All Documents"):
        delete_all_documents_from_collection()
        st.success("All documents have been deleted from the collection.")

if menu == "Show Documents in ChromaDB":
    st.subheader("Stored Documents in ChromaDB")
    documents = collection.get()["documents"]
    if documents:
        for i, doc in enumerate(documents, start=1):
            st.write(f"{i}. {doc}")
    else:
        st.write("No data available!")

elif menu == "Add New Document to ChromaDB as Vector":
    st.subheader("Add a New Document to ChromaDB")
    new_doc = st.text_area("Enter the new document:")
    uploaded_file = st.file_uploader("Or upload a .txt or .pdf file", type=["txt", "pdf"])

    if st.button("Add Document"):
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                if uploaded_file.type == "application/pdf":
                    temp_path = "/tmp/temp_pdf.pdf"
                    with open(temp_path, "wb") as temp_pdf:
                        temp_pdf.write(file_bytes)
                    articles = preprocess_pdf_constitution(temp_path)
                    for article in articles:
                        add_document_to_collection([article['text']], [article['id']])
                    st.success(f"Added {len(articles)} articles from the PDF successfully!")
                else:
                    detected_encoding = chardet.detect(file_bytes)['encoding']
                    if not detected_encoding:
                        raise ValueError("Failed to detect file encoding.")
                    file_content = file_bytes.decode(detected_encoding)

                    doc_id = f"doc{len(collection.get()['documents']) + 1}"
                    add_document_to_collection([file_content], [doc_id])
                    st.success(f"Document added successfully with ID {doc_id}")
            except Exception as e:
                st.error(f"Failed to add document: {e}")
        elif new_doc.strip(): 
            try:
                doc_id = f"doc{len(collection.get()['documents']) + 1}"
                add_document_to_collection([new_doc], [doc_id])
                st.success(f"Document added successfully with ID {doc_id}")
            except Exception as e:
                st.error(f"Failed to add document: {e}")
        else:
            st.warning("Please enter a non-empty document or upload a file before adding.")

if menu == "Ask Ollama a Question":
    query = st.text_input("Ask a question about the Constitution of Kazakhstan")
    if query:
        response, article_id = retrieve_and_answer(query, model)
        if response:
            st.write(f"Response: {response}")
        else:
            st.write("No response generated.")
        st.write(f"Cited Article/Section: {article_id}")


