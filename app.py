import streamlit as st
import openai
import os
from utils import load_and_index_pdfs, calculate_sharpe_ratio


import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


st.set_page_config(page_title="Finance Domain Chatbot", layout="wide")

st.title("Advanced Finance Domain Chatbot")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_and_index_pdfs()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a finance question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Deterministic logic trigger
    if "sharpe" in prompt.lower():
        sample_returns = [0.01, -0.005, 0.007, 0.012, -0.004]
        sharpe = calculate_sharpe_ratio(sample_returns)
        response = f"The calculated Sharpe Ratio is {sharpe}"
    
    else:
        docs = st.session_state.vector_store.similarity_search(prompt, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        system_prompt = f"""
        You are a professional financial analyst.
        Use only the context below.
        If answer not in context, say you don't know.

        Context:
        {context}
        """

        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        response = completion["choices"][0]["message"]["content"]

    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)
