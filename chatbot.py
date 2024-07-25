import PyPDF2
import streamlit as st


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import numpy as np

st.set_page_config(page_title= 'pdf_reader_chatbot')

st.markdown("""
            ## Following are the steps to follow: 
             
            1.**Enter your api key in provided text box** - You will need openai api key to acess the chatbot. For generating api key - https://platform.openai.com/settings/profile?tab=api-keys   

            2.**upload the documents in upload files section** 

            3.**click on submit button** 
            """)

api_key = st.text_input('enter openai api key:', type= 'password', key = "api_input")


def pdf_reader(docs_file):
    text = ""
    for pdf in docs_file:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def text_split(text):
    chunk = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap=100)
    text_chunk = chunk.split_text(text)
    return text_chunk

def embedded(text_chunks, api_key):
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding)
    vector_store.save_local("faiss_index")

def run_chain():
    prompt = """Answer the question as detail as possible from the input content. If Answer is not present in the content dont give wrong anwers. 
    content : {context}
    question : {question}
    answer :
"""
    model = ChatOpenAI(model_name = "gpt-3.5-turbo", openai_api_key = api_key)
    prompt_temp = PromptTemplate(template= prompt, input_variables= ['context', 'question'])
    chain = load_qa_chain(llm = model, chain_type= "stuff",prompt = prompt_temp)
    return chain
    
def user_input(user_question, api_key):
    embedding = OpenAIEmbeddings(model = "text-embedding-ada-002", openai_api_key= api_key)
    vector_load = FAISS.load_local("faiss_index", embedding,allow_dangerous_deserialization = True)
    doc = vector_load.similarity_search(user_question)
    chain = run_chain()
    response = chain({'input_documents':doc, "question": user_question}, return_only_outputs=True)
    st.write(response["output_text"])

def main():
    st.header('PDF READER CHATBOT')

    with st.sidebar:
        docs_file = st.file_uploader('upload multiple pdf files', accept_multiple_files= True, key = 'files')
        if st.button('submit', key= 'submit button') and api_key:
            with st.spinner('processing........'):
                pdf = pdf_reader(docs_file)
                split = text_split(pdf)
                embedded(split, api_key)
                st.success('done.')
    
    question = st.text_input('User question', key='question')


    response_container = st.empty()

    count = 0
    while question and api_key:
        response = user_input(question, api_key)
        response_container.write(response)

        count += 1
        question = st.text_input('User question', key=f'question{count}')
    
    
if __name__ == "__main__":
    main()