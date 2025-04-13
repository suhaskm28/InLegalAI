import streamlit as st
import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.preprocessing import normalize

# -------------------- Load Models & Index --------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
phi2_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", torch_dtype=torch.float16, device_map="auto"
)

faiss_index = faiss.read_index('vectorstore/legal_faiss_index.index')
with open('vectorstore/metadata.json', 'r') as f:
    metadata = json.load(f)

# -------------------- Helper Functions --------------------
def get_query_embedding(query):
    return normalize(embedding_model.encode([query]))

def search_vector(query_embedding, k=5):
    distances, indices = faiss_index.search(query_embedding, k)
    return distances, indices

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(phi2_model.device)
    outputs = phi2_model.generate(**inputs, max_new_tokens=300, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def retrieve_context(query):
    query_embedding = get_query_embedding(query)
    distances, indices = search_vector(query_embedding)
    chunks = []
    for idx in indices[0]:
        doc = metadata[idx]
        chunk = doc.get("chunks", [""])[0]
        title = doc.get("title", "")
        chunks.append(f"{title}: {chunk}")
    return "\n".join(chunks)

# -------------------- App Layout --------------------
st.set_page_config(page_title="Legal AI Assistant 🇮🇳", layout="wide")

# Top Nav
with st.container():
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 1rem; background-color: #f1f3f5; border-bottom: 1px solid #ccc;">
            <div style="font-size: 1.5rem;">🤖 Legal AI Assistant</div>
            <a href="?feature=Legal QA" style="text-decoration: none; font-weight: bold;">🏠 Legal QA</a>
        </div>
        """, unsafe_allow_html=True
    )

# Sidebar Navigation
with st.sidebar:
    st.image("logo.png", use_column_width=True)  # Add your logo file
    st.markdown("## 📚 Features")
    feature = st.radio("Navigate", ["Legal QA", "Document Analysis", "Case Analysis", "Complaint Drafting", "Other Legal Tasks"])

# -------------------- Feature Logic --------------------

# 1. Legal QA - Home Chat
if feature == "Legal QA":
    st.title("💬 Legal Question Answering")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.chat_input("Ask a legal question...")

    if user_query:
        # Add user input to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Generate context and answer
        context = retrieve_context(user_query)
        prompt = f"You are a helpful Indian Legal Assistant AI.\n\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"
        answer = generate_response(prompt)

        # Save assistant reply
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

# 2. Document Analysis
elif feature == "Document Analysis":
    st.title("📄 Document Analysis")
    uploaded = st.file_uploader("Upload a legal document (.txt or .pdf)", type=["txt", "pdf"])
    if uploaded:
        # TODO: Text Extraction, Chunking, Embedding, FAISS Querying
        st.info("🔧 Coming soon: Extraction, summarization, section mapping, etc.")

# 3. Case Analysis
elif feature == "Case Analysis":
    st.title("🔍 Case Analysis")
    case_query = st.text_input("Enter case or context:")
    if case_query:
        context = retrieve_context(case_query)
        prompt = f"Analyze the case: {case_query}\n\nContext:\n{context}\n\nAnswer:"
        answer = generate_response(prompt)
        st.markdown("#### 💡 Analysis")
        st.write(answer)

# 4. Complaint Drafting
elif feature == "Complaint Drafting":
    st.title("📝 Complaint Drafting")
    issue = st.text_area("Describe your issue:")
    if issue:
        prompt = f"Draft a legal complaint for the following issue in Indian context:\n{issue}\n\nComplaint:"
        draft = generate_response(prompt)
        st.markdown("#### 📄 Draft Complaint")
        st.write(draft)

# 5. Other Legal Tasks
elif feature == "Other Legal Tasks":
    st.title("🛠️ Other Legal Tasks")
    st.info("Add more tools here, like Notice Drafting, Contract Review, Legal Research Tools etc.")

