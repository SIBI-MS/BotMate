import streamlit as st
from chains import *



st.set_page_config(page_title='BotMate')
st.title("BotMate")
tab1, tab2, tab3 = st.tabs(["ChatwithPDF", "ChatwithWeb", "ChatwithVideo"])


with tab1:
    pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
    if st.button("Process"):
            with st.spinner("Loading.."):
                if "document_vector_store" not in st.session_state:  
                    document= get_document_data(pdf_docs) 
                    document_chunks=get_document_chunks(document)
                    embeddings=get_document_embedding()  
                    st.session_state.document_vector_store=get_vector_store(embeddings,document_chunks)
    
    if "document_vector_store" in st.session_state:
        user_query1 = st.chat_input("Ask your questions here...")


with tab2:
    pass
   


with tab3:
   pass
