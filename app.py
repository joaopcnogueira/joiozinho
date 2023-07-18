import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import streamlit as st
from streamlit_chat import message

pdf_file_path = 'pdf/compilatextosojoiodotrigofinal.pdf'
loader = PyPDFLoader(file_path=pdf_file_path)
pages = loader.load_and_split()

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())

chain = RetrievalQA.from_llm(llm = ChatOpenAI(temperature=0.0,
                                              model_name='gpt-4', 
                                              openai_api_key=os.getenv("OPENAI_API_KEY"), 
                                              max_tokens=256),
                             retriever = faiss_index.as_retriever())

def conversational_chat(query):
    awnswer = chain({"query": query})
    st.session_state['history'].append((query, awnswer["result"]))
    return awnswer["result"]

st.title("Joiozinho")
st.caption("Documentação em formato ChatBot da newsletter O Joio do Trigo")

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about Joio do Trigo 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]
    

response_container = st.container() # container for the chat history
container = st.container() # container for the user's text input

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

        print(st.session_state['history'])