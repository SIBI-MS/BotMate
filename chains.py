import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma



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
        
#Initialize the embedding
def get_document_embedding():
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf

#for store embeddings into vectordb using chroma
def get_vector_store(embeddings,document_chunks):
    db = Chroma.from_documents(document_chunks, embeddings)
    return db