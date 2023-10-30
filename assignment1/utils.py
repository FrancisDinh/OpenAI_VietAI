from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.docstore.document import Document


def str_to_list(input_str):
    # parse page_numbers to page number list
    if input_str:
        page_numbers = [int(page) for page in input_str.split(",")]
        return page_numbers
    return []


def read_pdf(file_path):
    # check file exist
    pdf_reader = PdfReader(file_path)
    return pdf_reader


def get_pdf_len(pdf_reader):
    return len(pdf_reader.pages)


def process_docs(pdf_reader):
    # Initialize an empty string to store the extracted text from the PDF
    text = ""

    # Iterate through the pages of the PDF and extract text from each page
    for i, page in enumerate(pdf_reader.pages):
        text += f"### Page {i}:\n\n" + page.extract_text()

    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="### ",
        chunk_overlap=200,
    )
    chunks = text_splitter.split_text(text)
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base


def search_doc_from_knowledge_base(knowledge_base, question, threshold=0.8):
    # Return the closet doc to question
    docs = knowledge_base.similarity_search_with_score(question)
    closest_doc = get_closet_doc_from_docs(
        docs, threshold=threshold)
    if closest_doc:
        return [closest_doc]
    return None


# Only return closest doc
def get_closet_doc_from_docs(docs, threshold):
    # Return the doc having the min score below threshold
    min_score = threshold
    min_id = -1
    for id, item in enumerate(docs):
        doc, score = item
        if min_score > score:
            min_id = id
            min_score = score
    if min_score < threshold:
        return docs[min_id][0]  # return doc
    else:
        return None


def get_pdf_len(pdf_reader):
    pdf_len = len(pdf_reader.pages)
    return pdf_len


def get_doc_pages(pdf_reader, pages):
    docs = []
    for page in pages:
        text = f" Page {page}:\n\n" + \
            pdf_reader.pages[page].extract_text()
        docs.append(Document(page_content=text))
    return docs
