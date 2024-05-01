import os
import streamlit as st
import pickle
import time
from PyPDF2 import PdfReader
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from prompt import *

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_url_text (urls):
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    text = ""
    for doc in docs:
        text += doc.page_content
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

file_path = "faiss_store_openai.pkl"

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

st.title("RetrieverBot: RAG Assisted Research Tool 📈")
option = st.sidebar.selectbox(
    'What would you like to process?',
    ('PDFs', 'URLs'))

if option == "URLs":
    st.sidebar.title("Enter the URLs:")
    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    process_clicked = st.sidebar.button("Process URLs")

elif option == "PDFs":
    st.sidebar.title("Enter your documents:")
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    process_clicked = st.sidebar.button("Process PDFs")

main_placeholder = st.empty()
llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.9, max_tokens=500, model="gpt-3.5-turbo-instruct")

if process_clicked:
    # load data
    main_placeholder.text("Data Loading...Started...✅✅✅")
    if option == "URLs":
        text = get_url_text(urls)
    elif option == "PDFs":
        text = get_pdf_text(pdf_docs)
    
    if text:
    # split data
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        chunks = get_text_chunks(text)

        if chunks:
            # create embeddings and save it to FAISS index
            main_placeholder.text("Embedding Vector Started Building...✅✅✅")
            vectorstore = get_vectorstore(chunks)
            time.sleep(2)

            # Save the FAISS index to a pickle file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)
        else:
            main_placeholder.text("Text Splitter produced empty chunks. Check data.")
    else:
        main_placeholder.text("Data loading failed. Check URLs/PDFs or network connection.")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore_openai = pickle.load(f)
            qa=RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=vectorstore_openai.as_retriever(search_kwargs={'k': 2}),
                return_source_documents=True, 
                chain_type_kwargs=chain_type_kwargs)

            result = qa({"query": query})
            st.header("Answer")
            st.write(result["result"])