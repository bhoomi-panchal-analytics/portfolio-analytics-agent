import os
import faiss
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_and_index_pdfs(folder_path="data"):
    documents = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(texts, embeddings)

    return vector_store


# Deterministic Financial Module
def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate/252
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return round(sharpe * np.sqrt(252), 4)
