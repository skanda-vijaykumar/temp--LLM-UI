import gradio as gr
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
import os
import timeit 
# Load the PDF documents
loader = DirectoryLoader(r'C:\Users\Skanda\Desktop\INTERNSHIP\llama2\pdf',
                         glob="*.pdf",
                         loader_cls=PyPDFLoader)
documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                               chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Create the embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})

# Create the vector store
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local('vectorstore/db_faiss')

# Set up the QA prompt
qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Load the LLM
from langchain.llms import CTransformers

llm = CTransformers(model=r'C:\Users\Skanda\Desktop\INTERNSHIP\llama2\llama-2-7b-chat.Q8_0.gguf', 
                    model_type='llama',                    
                    config={'max_new_tokens': 500,
                            'context_length' : 4096,
                            'temperature': 0.01})

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt

def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k':2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt})
    return dbqa

def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa

# Gradio UI
def run_gradio():
    dbqa = setup_dbqa()

    def answer_question(query):
        start = timeit.default_timer()
        response = dbqa({'query': query})
        end = timeit.default_timer()

        result = response["result"]
        source_docs = response['source_documents']

        source_text = ""
        for i, doc in enumerate(source_docs):
            source_text += f'\nSource Document {i+1}\n'
            source_text += f'Source Text: {doc.page_content}\n'
            source_text += f'Document Name: {doc.metadata["source"]}\n'
            source_text += f'Page Number: {doc.metadata["page"]+1}\n'
            source_text += '='* 50 + '\n'

        return result, source_text, end - start

    with gr.Blocks() as demo:
        gr.Markdown("# Question Answering")
        with gr.Row():
            with gr.Column():
                query = gr.Textbox(label="Question")
                submit = gr.Button("Submit")
            with gr.Column():
                result = gr.Textbox(label="Answer", interactive=False)
                source_text = gr.Textbox(label="Source Documents", interactive=False, lines=10)
                time_taken = gr.Textbox(label="Time Taken", interactive=False)

        submit.click(answer_question, inputs=query, outputs=[result, source_text, time_taken])

    demo.launch()

if __name__ == "__main__":
    run_gradio()