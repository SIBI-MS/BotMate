import streamlit as st
from chains import *



st.set_page_config(page_title='BotMate')
st.title("BotMate")
tab1, tab2, tab3 = st.tabs(["ChatwithPDF", "ChatwithWeb", "ChatwithVideo"])


with tab1:
    pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
    if st.button("Process"):
            with st.spinner("Loading.."):
                if "vector_store" not in st.session_state:  
                    document= get_documentdata(pdf_docs) 
                    document_chunks=get_document_chunks(document)
                    embeddings=get_document_embedding()  
                    st.session_state.document_vector_store=get_vector_store(embeddings,document_chunks)
                    # vector_store=get_store_embeddings(document_embedding)              
    
    if pdf_docs:
        user_query1 = st.chat_input("Type your message here...")


with tab2:
    pass
   


with tab3:
   pass
