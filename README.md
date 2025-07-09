# 🏦 Loan Dataset Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that intelligently answers questions about a loan approval dataset using document retrieval and a lightweight language model.

---

## 📌 Project Overview

This chatbot uses:
- 🔎 **Document Retrieval** with sentence embeddings + FAISS
- 🤖 **Generative LLMs** (e.g., `flan-t5-base` from Hugging Face) for response generation
- 🧠 Your cleaned **loan dataset** as a knowledge base
- 🧪 Built in **Colab**, deployable via **Streamlit** or **Hugging Face Spaces**

---

## 📁 Dataset

Dataset source: [Kaggle - Loan Approval Prediction](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction)

> Ensure that your dataset is cleaned (no nulls, all numerical and categorical fields handled).

---

## Streamlit app link
https://loanragqnabot.streamlit.app/

---

## 🔧 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/loan-rag-chatbot.git
cd loan-rag-chatbot
