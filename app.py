import streamlit as st
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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

# Helper Functions
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

# Streamlit UI
st.title("Legal AI Assistant")
option = st.sidebar.selectbox("Choose a Feature", ["Case Search", "Legal Question Answering"])

if option == "Case Search":
    st.header("Search for Legal Cases")
    query = st.text_input("Enter your legal query:")
    
    if query:
        st.write(f"Searching for: **{query}**")
        
        # Get embedding for the query
        query_embedding = get_query_embedding(query)
        
        # Perform FAISS search
        distances, indices = search_vector(query_embedding)
        
        # Display results
        for i, idx in enumerate(indices[0]):
            case_metadata = metadata[idx]
            st.subheader(f"{case_metadata['title']} ({case_metadata['date']})")
            st.write(f"**Court**: {case_metadata.get('court', 'N/A')}")
            st.write(f"**Sections Mentioned**: {case_metadata.get('sections', 'N/A')}")
            st.write(f"**Citations**: {case_metadata.get('citations', 'N/A')}")
            st.write(f"**Case Summary**: {case_metadata.get('chunks', ['No content available'])[0]}")
            st.write(f"**Similarity Score**: {distances[0][i]}")
            st.markdown("---")

elif option == "Legal Question Answering":
    st.header("Ask a Legal Question")
    question = st.text_input("Enter your legal question:")
    
    if question:
        st.write(f"Getting answer for: **{question}**")
        
        # Get embedding for the question
        query_embedding = get_query_embedding(question)
        
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
        prompt = f"You are a legal assistant AI. Based on the following legal cases:\n{context}\n\nAnswer the following legal question:\n{question}\n\nAnswer:"
        answer = generate_response(prompt)
        
        # Display the answer
        st.subheader("Answer:")
        st.write(answer)
