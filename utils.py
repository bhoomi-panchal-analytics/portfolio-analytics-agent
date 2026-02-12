import os
import numpy as np
from pypdf import PdfReader


def load_and_index_pdfs(folder_path="data"):
    documents = []

    if not os.path.exists(folder_path):
        return []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder_path, file))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            documents.append(text)

    return documents


def simple_search(documents, query):
    results = []
    for doc in documents:
        if query.lower() in doc.lower():
            results.append(doc[:1000])  # return first 1000 chars
    return results[:3]


def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return round(sharpe * np.sqrt(252), 4)
