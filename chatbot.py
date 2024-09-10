import langchain
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
import google.generativeai as Genai 
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
import numpy as np

st.set_page_config(page_title= 'pdf_reader_chatbot')

st.markdown("""
<style>
.st-emotion-cache-15ecox0.ezrtsby0          
{
    visibility:hidden;
}
</style>""")

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
    embedded = GoogleGenerativeAIEmbeddings(model = "models/embedding-001",google_api_key = api_key)
    vector_store = FAISS.from_texts(chunk, embedding=embedded)
    vector_store.save_local("faiss_index")

def model(api_key):
    prompt = """Answer the question as detail as possible from the input content. If Answer is not present in the content dont give wrong anwers. 
    content : \n{context} \n
    question : \n{question} \n    answer :
"""
    model = ChatGoogleGenerativeAI(model = "gemini-pro", google_api_key = api_key)
    prompt_temp = PromptTemplate(template= prompt, input_variables= ['context', 'question'])
    chain = load_qa_chain(llm = model, chain_type= "stuff",prompt = prompt_temp)
    return chain

def run_chain(user_question, api_key):
    embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key= api_key)
    vector_load = FAISS.load_local("faiss_index", embedding,allow_dangerous_deserialization = True)
    doc = vector_load.similarity_search(user_question)
    chain = model(api_key)
    response = chain({'input_documents':doc, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


def main():
    st.header('PDF READER CHATBOT')
    
    with st.sidebar:
        st.header("Menu:")
        api_key = 'AIzaSyDZXCmww99wIDGiijYJXV5W6WlxKtdoeWc'          #st.text_input('enter openai api key:', type= 'password', key = "api_input")
        docs_file = st.file_uploader('upload pdf files:', accept_multiple_files= True, key = 'files')
        if st.button('submit', key= 'submit button') and api_key:
            with st.spinner('processing........'):
                pdf = extract_pdf(docs_file)
                split = text_splitter(pdf)
                embedding(split, api_key)
                st.success('done.')

    if "messages" not in st.session_state:
        st.session_state.messages = []
      
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("ask something"):
        if api_key:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role":"user", "content": prompt}) 

            if prompt and api_key:
                response = run_chain(user_question = prompt, api_key = api_key)
            
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                st.session_state.messages.append({"role":"assistant", "content": response}) 

        else:
            st.error("enter api key first")


if __name__ == "__main__":
    main()