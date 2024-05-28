import streamlit as st
from InitialProcess import *
from Retriever import *


st.set_page_config(page_title='BotMate')
st.title("BotMate")
tab1, tab2, tab3 = st.tabs(["ChatwithPDF", "ChatwithWeb", "ChatwithVideo"])


with tab1:
    pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
    if st.button("Process"):
            with st.spinner("Loading.."):
                if "vector_store" not in st.session_state:  
                    document= get_document_data(pdf_docs) 
                    document_chunks=get_document_chunks(document)
                    embeddings=get_document_embedding()  
                    document_vector_store=get_vector_store(embeddings,document_chunks)
                    st.session_state.vector_store = document_vector_store
                    saver(document_vector_store)
    
    if "vector_store" in st.session_state:
        user_query = st.chat_input("Ask your questions here...")
    invoker(user_query)



with tab2:
    pass
   


with tab3:
   pass
