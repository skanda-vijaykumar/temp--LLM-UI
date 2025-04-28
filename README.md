## Gradio PDF Question-Answering Chatbot

**Key Functionality**  
- Ingests all PDF files from a specified directory using LangChain’s `DirectoryLoader` and `PyPDFLoader`  
- Splits each document into overlapping text chunks (`chunk_size=500`, `chunk_overlap=50`)  
- Generates embeddings for every chunk with the `all-MiniLM-L6-v2` sentence-transformers model  
- Builds and persists a FAISS vector store (`vectorstore/db_faiss`) for efficient similarity search  
- Defines a custom QA prompt template that conditions answers strictly on retrieved context  
- Loads a local Llama 2 model via `CTransformers` for answer generation  
- Constructs a RetrievalQA chain that returns both the model’s answer and its source documents  
- Launches a Gradio interface where users submit questions and receive:  
  - The concise answer  
  - Extracted source snippets with document names and page numbers  
  - Query processing time  



**Usage**  
1. Configure the script’s paths:  
   - Point `DirectoryLoader` to your PDF folder.  
   - Set the `CTransformers` model path to your local Llama 2 GGUF file.  

2. Run the script:  
   ```bash
   python your_script.py
   ```
3. In the browser window that opens, enter your question, click **Submit**, and view the answer along with source excerpts and timing.

