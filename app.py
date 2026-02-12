import os
import streamlit as st
from openai import OpenAI
from utils import load_and_index_pdfs, calculate_sharpe_ratio

# ----------------------------
# Configuration
# ----------------------------

st.set_page_config(page_title="Finance Intelligence Agent", layout="wide")
st.title("Finance Intelligence Agent")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# ----------------------------
# Initialize Vector Store
# ----------------------------

if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = load_and_index_pdfs()
    except Exception as e:
        st.error(f"Vector store loading failed: {e}")
        st.stop()

# ----------------------------
# Chat Memory
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# User Input
# ----------------------------

prompt = st.chat_input("Ask a finance-related question")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # ----------------------------
    # Deterministic Trigger (Sharpe Ratio)
    # ----------------------------

    if "sharpe" in prompt.lower():

        sample_returns = [0.01, -0.005, 0.007, 0.012, -0.004]
        sharpe = calculate_sharpe_ratio(sample_returns)
        response_text = f"The calculated Sharpe Ratio is {sharpe}"

    else:

        # Retrieve relevant documents
        docs = st.session_state.vector_store.similarity_search(prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        system_prompt = f"""
You are an institutional-grade financial analyst.

Use only the provided context.
If the answer is not in the context, say: "Insufficient data in provided documents."

Context:
{context}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            response_text = response.choices[0].message.content

        except Exception as e:
            response_text = f"Model error: {e}"

    # ----------------------------
    # Display Assistant Response
    # ----------------------------

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )

    with st.chat_message("assistant"):
        st.markdown(response_text)
