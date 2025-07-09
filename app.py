import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load cleaned data
df_clean = pd.read_csv("cleaned_loan_data.csv")

# Preprocess rows into text
def row_to_text(row):
    return " | ".join([f"{col}: {row[col]}" for col in df_clean.columns])

# Prepare documents and FAISS index
@st.cache_resource
def setup_retrieval():
    documents = df_clean.apply(row_to_text, axis=1).tolist()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    return embedder, index, documents

embedder, index, documents = setup_retrieval()

# Load lightweight LLM
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

llm = load_llm()

# Function to retrieve documents
def retrieve_docs(query, k=5):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [documents[i] for i in I[0]]

# Generate answer using LLM
def generate_answer(query, context_docs):
    context = "\n".join(context_docs)
    prompt = f"""Answer the question based on the following context:

{context}

Question: {query}
Answer:"""
    result = llm(prompt, max_new_tokens=256)[0]["generated_text"]
    return result.strip()

# Streamlit UI
st.title("Loan Dataset Q&A Chatbot 💬")
user_input = st.text_input("Ask a question about the loan dataset:")

if user_input:
    with st.spinner("Thinking..."):
        context = retrieve_docs(user_input)
        answer = generate_answer(user_input, context)
    st.markdown("**Answer:**")
    st.write(answer)
