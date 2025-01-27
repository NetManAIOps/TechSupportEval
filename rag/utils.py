import os
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

CACHE_DIR = Path(os.path.dirname(__file__)) / '../tmp/rag/'

TECHQA_PATH = Path(os.path.dirname(__file__)) / '../data/rag_inputs/techqa.json'

TEST_QUESTION_ID = 3

def langchain_load_doc(reference_doc: str | list[str], splitter=None):
    docs = [reference_doc] if isinstance(reference_doc, str) else reference_doc
    documents = [Document(page_content=doc) for doc in docs]
    if splitter is None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def langchain_create_vectorstore(name: str, question_id: str, reference_doc: str | list[str], splitter=None):
    cache_path = CACHE_DIR / name / f'{question_id}'

    documents = langchain_load_doc(reference_doc, splitter)
    embeddings = OpenAIEmbeddings()

    if os.path.exists(cache_path):
        return FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)

    vectorstore = FAISS.from_documents(documents, embeddings)
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR / name, exist_ok=True)

    vectorstore.save_local(cache_path)

    return vectorstore


def get_test_question() -> dict:
    with open(TECHQA_PATH, 'r') as f:
        data = json.load(f)
    return data[TEST_QUESTION_ID]