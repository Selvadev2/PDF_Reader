import langchain
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_community.embeddings import OpenAIEmbeddings
import google.generativeai as Genai 
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title= 'pdf_reader_chatbot', layout= "wide")

st.markdown("""
<style>
.st-emotion-cache-15ecox0.ezrtsby0          
{
    visibility: hidden;
}
.styles_terminalButton__JBj5T
{
    visibility: hidden;
}          
</style>""", unsafe_allow_html= True)

# extract all the pdfs
def extract_pdf(files):
    text = ''
    for file in files:
        read = PyPDF2.PdfReader(file)
        for i in read.pages:
            text += (i.extract_text()) 
    return text


# convert text to chunks
def text_splitter(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 100)
    chunk = splitter.split_text(text)
    return chunk

def embedding(chunk, api_key):
    embedded = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004",google_api_key = api_key)
    vector_store = FAISS.from_texts(chunk, embedding=embedded)
    vector_store.save_local("faiss_index_google")

def embedding_openai(chunk, api_key_openai):
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key_openai)
    vector_store = FAISS.from_texts(chunk, embedding=embedding)
    vector_store.save_local("faiss_index_openai")


def model(api_key):
    prompt = """Answer the question as detail as possible from the input content. If Answer is not present in the content dont give wrong anwers. 
    content : \n{context} \n
    question : \n{question} \n    answer :
   """
    model = ChatGoogleGenerativeAI(model = "gemini-pro", google_api_key = api_key)
    prompt_temp = PromptTemplate(template= prompt, input_variables= ['context', 'question'])
    chain = load_qa_chain(llm = model, chain_type= "stuff",prompt = prompt_temp)
    return chain

def model_openai(api_key_openai):
    prompt = """Answer the question as detail as possible from the input content. If Answer is not present in the content dont give wrong anwers. 
    content : {context}
    question : {question}
    answer :
"""
    model = ChatOpenAI(model_name = "gpt-3.5-turbo", openai_api_key = api_key_openai)
    prompt_temp = PromptTemplate(template= prompt, input_variables= ['context', 'question'])
    chain = load_qa_chain(llm = model, chain_type= "stuff",prompt = prompt_temp)
    return chain

def run_chain(user_question, api_key_google):
    embedded = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004", google_api_key= api_key_google)
    vector_load = FAISS.load_local("faiss_index_google", embedded,allow_dangerous_deserialization = True)
    doc = vector_load.similarity_search(user_question)
    chain = model(api_key_google)
    response = chain({'input_documents':doc, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def run_chain_openai(user_question, api_key_openai):
    embedding = OpenAIEmbeddings(model = "text-embedding-ada-002", openai_api_key= api_key_openai)
    vector_load = FAISS.load_local("faiss_index_openai", embedding,allow_dangerous_deserialization = True)
    doc = vector_load.similarity_search(user_question)
    chain = model_openai(api_key_openai)
    response = chain({'input_documents':doc, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.header('PDF READER CHATBOT')
    
    with st.sidebar:
        st.header("Menu:")
        docs_file = st.file_uploader('upload pdf files:', accept_multiple_files= True, key = 'files')
        model_selection = st.radio("select the model:", options = ['gemini','chatgpt'])
            
        api_key_google = st.secrets["api_key_g"] # os.getenv("api_key_g")
        api_key_openai = st.secrets["api_key_o"] # os.getenv("api_key_o")


        if st.button('submit', key= 'submit button') and docs_file:
            with st.spinner('processing........'):
                pdf = extract_pdf(docs_file)
                split = text_splitter(pdf)
                if model_selection == 'gemini':
                    embedding(split, api_key_google)
                else:
                    embedding_openai(split, api_key_openai)
                st.success('done.')

    if "messages" not in st.session_state:
        st.session_state.messages = []
      
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("ask something"):
        
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role":"user", "content": prompt}) 

        if prompt:
            if model_selection == 'gemini':
                response = run_chain(user_question = prompt, api_key_google = api_key_google)
            else:
                response = run_chain_openai(user_question = prompt, api_key_google = api_key_openai)

        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role":"assistant", "content": response}) 


if __name__ == "__main__":
    main()