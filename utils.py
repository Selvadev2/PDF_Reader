import langchain
import PyPDF2
import pytesseract
from pdf2image import convert_from_path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_openai import OpenAIEmbeddings
import google.generativeai as Genai 
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
import numpy as np 
import os
import tempfile


# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


# extract all the pdfs
def extract_pdf(files, language):
    dic = {'English':'eng','Marathi':'mar', 'Hindi':'hin', 'Gujarati':'guj'}
    if language != 'English':
        language_str = 'eng+' + dic[language]
    else:
        language_str = 'eng'
    text = ''
    for uploaded_file in files:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            pages = convert_from_path(temp_file_path, dpi= 200 )#poppler_path= "C:/Users/DIPL/poppler-24.07.0/Library/bin"
            for i in pages:
                text += pytesseract.image_to_string(i, lang= language_str) 
        os.remove(temp_file_path)
    return text


# convert text to chunks
def text_splitter(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap= 100)
    chunk = splitter.split_text(text)
    return chunk



def embedding( chunk, api_key):
    embedded = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004",google_api_key = api_key)
    vector_store = FAISS.from_texts(chunk, embedding=embedded)
    vector_store.save_local("faiss_index_google")

def model(api_key):
    prompt = """
    Design a PDF document extractor that filters and extracts only tender-related documents specific to the paper printing industry. 
    The extractor should identify keywords and phrases like 'printing services,' 'paper procurement,' 'security printing,' 'supply of paper materials,' 
    and 'request for quotations (RFQ)' for printing or related services. It should capture critical details, such as:

    - Penalties for Delay
    - Pre Qualification Criteria
    - Due date & Last offline tender submission date
    - Terms and Conditions of tender
    - Delivery Time and Place ,Packing details
    - Tender Fee EMD And Security Deposit
    - Financial Turnover Requirement for Bidders
    - Specifications
    - Required Documents for Bidders
    - Address
    - Specification
    - "Delivery Place,Contact Details,Contact Details,Pre-bid Meeting,Estimated Cost,
       IBA Required,Summarize:,Required Document,Sample Printing Details"
    - Tender Schedule
    The tool should also process documents in both English and {language}, ensuring accurate extraction even when documents contain mixed languages.
    if answer is not present in the pdf don't give wrong answers type 'answer not found in pdf'. 
    if the user has provided an unclear or vague question: "{question}".
    Please interpret rephrased question to clarify its meaning and then answer accordingly.
    Content: 
    {context} 

    Question (in {language} or English): 
    {question} 

    Answer (in {language} and in English):

    """
    model = ChatGoogleGenerativeAI(model = "gemini-pro", google_api_key = api_key)
    prompt_temp = PromptTemplate(template= prompt, input_variables= ['context', 'question', 'language'])
    chain = load_qa_chain(llm = model, prompt = prompt_temp)
    return chain

def run_chain(user_question, language, api_key_google):
    embedded = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004", google_api_key= api_key_google)
    vector_load = FAISS.load_local("faiss_index_google", embedded,allow_dangerous_deserialization = True)
    doc = vector_load.similarity_search(user_question)
    chain = model(api_key_google)
    response = chain.invoke({'input_documents':doc, "question": user_question, 'language': language}, return_only_outputs=True)
    return response["output_text"]


def embedding_openai(chunk, api_key_openai):
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key_openai)
    vector_store = FAISS.from_texts(chunk, embedding=embedding)
    vector_store.save_local("faiss_index_openai")

def model_openai(api_key_openai):
    prompt = """
    Design a PDF document extractor that filters and extracts only tender-related documents specific to the paper printing industry. 
    The extractor should identify keywords and phrases like 'printing services,' 'paper procurement,' 'security printing,' 'supply of paper materials,' 
    and 'request for quotations (RFQ)' for printing or related services. It should capture critical details, such as:

    - Penalties for Delay
    - Pre Qualification Criteria
    - Due date & Last offline tender submission date
    - Terms and Conditions of tender
    - Delivery Time and Place ,Packing details
    - Tender Fee EMD And Security Deposit
    - Financial Turnover Requirement for Bidders
    - Specifications
    - Required Documents for Bidders
    - Address
    - Specification
    - "Delivery Place,Contact Details,Contact Details,Pre-bid Meeting,Estimated Cost,
       IBA Required,Summarize:,Required Document,Sample Printing Details"
    - Tender Schedule
    The tool should also process documents in both English and {language}, ensuring accurate extraction even when documents contain mixed languages.
    if answer is not present in the pdf don't give wrong answers type 'answer not found in pdf'. 
    if the user has provided an unclear or vague question: "{question}".
    Please interpret rephrased question to clarify its meaning and then answer accordingly.
    Content: 
    {context} 

    Question (in {language} or English): 
    {question} 

    Answer (in {language} and in English):
    """
    model = ChatOpenAI(model_name = "gpt-3.5-turbo-0125", openai_api_key = api_key_openai)
    prompt_temp = PromptTemplate(template= prompt, input_variables= ['context', 'question', 'language'])
    chain = load_qa_chain(llm = model, prompt = prompt_temp)
    return chain
    
def run_chain_openai(user_question, language, api_key_openai, index_file="faiss_index_openai"):
    embedding = OpenAIEmbeddings(model = "text-embedding-ada-002", openai_api_key= api_key_openai)
    vector_load = FAISS.load_local("faiss_index_openai", embedding,allow_dangerous_deserialization = True)
    doc = vector_load.similarity_search(user_question)
    chain = model_openai(api_key_openai)
    response = chain.invoke({'input_documents':doc, "question": user_question, 'language': language}, return_only_outputs=True)
    return response["output_text"]

