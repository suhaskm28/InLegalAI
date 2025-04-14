import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import torch
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
    st.session_state.chat_history.append({"user": query, "ai": "Generating answer..."})

    # Get embedding for the question
    query_embedding = get_query_embedding(query)

    # Perform FAISS search (top 3 for optional compression)
    distances, indices = search_vector(query_embedding, k=3)

    # Use only top-1 for direct answer, but gather more if needed
    doc_indices = indices[0]
    top_doc = metadata[doc_indices[0]]
    top_title = top_doc.get("title", "")
    top_chunks = top_doc.get("chunks", [])

    # Optional: RAG-style compression (combine if short enough)
    combined_chunk = ""
    for c in top_chunks[:2]:  # Use top 2 chunks if available
        if len(combined_chunk) + len(c) < 1200:
            combined_chunk += c + "\n"
        else:
            break

    if not combined_chunk.strip():
        combined_chunk = top_chunks[0] if top_chunks else ""

    # Build final prompt
    prompt = f"""You are a helpful and knowledgeable legal assistant AI.

Below is an excerpt from a legal judgment or case document.

Use it to answer the user's question clearly and understandably. Do not copy verbatim unless quoting is necessary. If the text does not contain the answer, politely explain that.

Legal Document (Case: {top_title}):
{combined_chunk.strip()}

User's Question:
{query}

Answer:"""

    # Generate answer using Phi-2
    answer = generate_response(prompt)

    # Update chat history
    st.session_state.chat_history[-1]["ai"] = answer

    # Show traceability / source
    st.session_state.chat_history[-1]["source"] = top_title


# Display chat history
for item in st.session_state.chat_history:
    st.chat_message("user").write(item["user"])
    st.chat_message("assistant").write(item["ai"])
    if "source" in item:
        st.markdown(f"<div style='font-size: 0.85em; color: gray;'>📄 Source: *{item['source']}*</div>", unsafe_allow_html=True)
