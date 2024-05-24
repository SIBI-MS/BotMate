import fitz  # PyMuPDF



def get_documentdata(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        document = fitz.open(pdf)
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()

    document.close()