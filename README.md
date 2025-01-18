# advprog3  
Assignment 3 Advanced Programming. Teamwork by Togzhan Oral and Yelnura Akhmetova.    

## Overview  

This project integrates **Streamlit**, **ChromaDB**, and **OllamaLLM** to enable document retrieval and interactive Q&A functionality. The application is specifically designed to interact with the **Constitution of Kazakhstan**, providing a way to query specific articles or sections of the document and retrieve answers. The application supports:  
- Uploading documents to ChromaDB with embeddings generated using SentenceTransformer.  
- Querying documents for context and providing answers through Ollama.  


## Installation  

1. Clone the repository:  
    ```bash  
    git clone https://github.com/yourusername/chat-with-ollama.git  
    ```  

2. Navigate to the project directory:  
    ```bash  
    cd chat-with-ollama  
    ```  

3. Install the dependencies:  
    ```bash  
    pip install -r requirements.txt  
    ```  

4. Ensure API keys and environment variables are set up (if needed).  

## Usage  

1. Start the Streamlit application:  
    ```bash  
    streamlit run src/app.py  
    ```  

2. Use the app features through your browser:  
    - **Show Documents**: View all documents stored in ChromaDB.  
    - **Add Document**: Upload a new document as a text input or file, and store it in ChromaDB.  
    - **Ask a Question**: Query the stored documents and receive answers via Ollama.  

## Features  

### Add Documents  
- Add a document through text input or by uploading a `.txt` or `.pdf` file.  
- Each document is embedded into a vector representation using SentenceTransformer and stored in ChromaDB.  

### Show Documents  
- View all documents currently stored in ChromaDB.  

### Query and Answer  
- Input a question to retrieve relevant documents from ChromaDB.  
- Generate an answer by querying Ollama with the retrieved document context.

### Constitution of Kazakhstan  
- The application allows users to interact with the **Constitution of Kazakhstan** specifically.  
- You can query specific articles or sections of the Constitution and receive answers based on the retrieved document context.  
- The app processes the uploaded Constitution (in `.txt` or `.pdf` format) and stores its articles as separate documents in ChromaDB for easy retrieval.


## Code Highlights  

### Document Addition  
- Handles both manual text input and `.txt`/`.pdf` file uploads.  
- Automatically detects the encoding of uploaded files to ensure proper handling.  
- Stores embeddings in ChromaDB using `SentenceTransformer`.  

### Query with Contextual Augmentation  
- Retrieves relevant documents using a similarity query.  
- Augments the user query with retrieved document context before passing it to Ollama.  

### Error Handling  
- Logs issues with embedding generation or API queries.  
- Ensures robustness for invalid inputs or empty documents.  

## Example Workflow  

1. **Add a New Document**  
    - Use the text area or file upload option to add a document.  

2. **Ask a Question**  
    - Type a question in the input field.  
    - The app retrieves the most relevant document and generates an answer using Ollama.  

## Screenshot of the Interface  

<img width="1436" alt="image" src="https://github.com/user-attachments/assets/f78f3bfe-6634-48d0-93b9-3af41bcb4bc0" />
