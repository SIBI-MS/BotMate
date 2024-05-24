import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter


#for extract data from the documents
def get_documentdata(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        document = fitz.open(pdf)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
    document.close()
    return text


#for making chunks
def get_document_chunks(document):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    document_chunks = text_splitter.create_documents(document)
    return document_chunks
        
