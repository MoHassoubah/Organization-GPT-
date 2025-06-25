from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
# import nltk
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from typing import List
from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings
# from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


CHROMA_PATH = "D:/scripts/seitech_gpt/chroma"


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
        # region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings



folder_path="test resources"


docx_DATA_PATH = "D:/scripts/seitech_gpt/data/docx"
pdf_DATA_PATH = "D:/scripts/seitech_gpt/data/pdfs"
def load_pdf_documents():
    document_loader = PyPDFDirectoryLoader(pdf_DATA_PATH)
    return document_loader.load()
    
def load_docx_documents():
    document_loader = DirectoryLoader(docx_DATA_PATH, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
    return document_loader.load()
    

def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
    
def add_to_chroma(chunks: List[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)

    else:
        print("✅ No new documents to add")
    return db


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        # page = chunk.metadata.get("page")
        # current_page_id = f"{source}:{page}"
        current_page_id = f"{source}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks
    
    
    
import requests
import json
import time

def generate_response(prompt, model='llama2'):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    print(response)
    return json.loads(response.text)['response']

import torch

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print("CUDA optimizations enabled...")
else:
    print("CUDA is not available")
    
# query_text = "give a name and brief descirption of the content all the documents found in the database" 
# query_text = "list me 3 possible candidates  names for AI engineer position from the avaliable candidates without hullicinating any names" 
docs = load_docx_documents()
docs.extend(load_pdf_documents())
# print(docs)
chunks = split_documents(docs)
print(len(chunks))
db = add_to_chroma(chunks)

while(True):
    query_text = input(">Ask: ")
    if query_text == 'exit':
        print("thank you :)")
        break
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    start_time = time.time()
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    end_time = time.time()
    # print(f'langchain time {end_time - start_time}')


    # start_time = time.time()
    # # Example usage
    response = generate_response(prompt)
    # end_time = time.time()

    # print(f'time url {end_time - start_time}')
    # print(response)

    # model = Ollama(model="llama2")
    # start_time = time.time()
    # response = model.invoke(prompt)
    # end_time = time.time()
    # print(f'second inove time{end_time-start_time}')

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f">>Response: {response}\nSources: {sources}"
    print(formatted_response)
    print(" ")
    print(" ")
