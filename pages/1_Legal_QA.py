import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
import json
from sklearn.preprocessing import normalize

# Load FAISS index and metadata
faiss_index = faiss.read_index('vectorstore/legal_faiss_index.index')
with open('vectorstore/metadata.json', 'r') as f:
    metadata = json.load(f)

# Embedding model (same as used for vector DB creation)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Phi-2 from HuggingFace
phi2_model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(phi2_model_id)
phi2_model = AutoModelForCausalLM.from_pretrained(
    phi2_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Helper Functions for FAISS and Generation
def get_query_embedding(query):
    """Convert a query into its vector representation."""
    embedding = embedding_model.encode([query])
    return normalize(embedding)

def search_vector(query_embedding, k=5):
    """Perform vector search in FAISS."""
    distances, indices = faiss_index.search(query_embedding, k)
    return distances, indices

def generate_response(prompt):
    """Generate a response using the Phi-2 model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(phi2_model.device)
    outputs = phi2_model.generate(**inputs, max_new_tokens=300, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Legal QA Chat Interface
st.title("💬 Legal Question Answering")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask a legal question...")

if query:
    st.session_state.chat_history.append({"user": query, "ai": "Answer coming soon..."})  # Replace with actual response

    # Get embedding for the question
    query_embedding = get_query_embedding(query)

    # Perform FAISS search
    distances, indices = search_vector(query_embedding)

    # Collect context from the results
    retrieved_chunks = []
    for i, idx in enumerate(indices[0]):
        doc = metadata[idx]
        chunk = doc.get("chunks", [""])[0]
        title = doc.get("title", "")
        retrieved_chunks.append(f"{title}: {chunk}")

    # Build context
    context = "\n".join(retrieved_chunks)

    # Generate response using Phi-2
    prompt = f"You are a legal assistant AI. Based on the following legal cases:\n{context}\n\nAnswer the following legal question:\n{query}\n\nAnswer:"
    answer = generate_response(prompt)

    # Update chat with answer
    st.session_state.chat_history[-1]["ai"] = answer

# Display chat history
for item in st.session_state.chat_history:
    st.chat_message("user").write(item["user"])
    st.chat_message("assistant").write(item["ai"])
