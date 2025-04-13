import streamlit as st

# Configure the Streamlit page
st.set_page_config(page_title="LegalAI 🇮🇳", page_icon="⚖️", layout="wide")

# Inject custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #283e4a;
        margin-top: 1rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #4b4b4b;
        margin-bottom: 1.5rem;
    }
    .feature-box {
        border: 1px solid #dedede;
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }
    .feature-box:hover {
        background-color: #e2ecf5;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("assets/logo.png", width=150)
    st.markdown("## 🧠 Features")
    st.page_link("pages/1_Legal_QA.py", label="🏠 Legal QA", icon="💬")
    st.page_link("pages/2_Document_Analysis.py", label="📄 Document Analysis", icon="📘")
    st.page_link("pages/3_Case_Analysis.py", label="⚖️ Case Analysis", icon="📚")
    st.page_link("pages/4_Complaint_Drafting.py", label="📝 Complaint Drafting", icon="✍️")

# Main content
st.markdown("<div class='main-title'>Welcome to LegalAI 🇮🇳</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your personal AI legal assistant for Indian case insights, document analysis, and legal drafting.</div>", unsafe_allow_html=True)

# Feature Grid
col1, col2 = st.columns(2)

with col1:
    if st.button("💬 Legal QA"):
        st.switch_page("pages/1_Legal_QA.py")
    st.markdown("<div class='feature-box'>Ask any legal question and get case-backed AI answers in real-time.</div>", unsafe_allow_html=True)

    if st.button("📘 Document Analysis"):
        st.switch_page("pages/2_Document_Analysis.py")
    st.markdown("<div class='feature-box'>Upload legal documents and receive intelligent section-wise summaries and insights.</div>", unsafe_allow_html=True)

with col2:
    if st.button("📚 Case Analysis"):
        st.switch_page("pages/3_Case_Analysis.py")
    st.markdown("<div class='feature-box'>Explore case-wise breakdowns, similar judgments, and RAG-based query analysis.</div>", unsafe_allow_html=True)

    if st.button("✍️ Complaint Drafting"):
        st.switch_page("pages/4_Complaint_Drafting.py")
    st.markdown("<div class='feature-box'>Draft professional legal complaints tailored to your case, in seconds.</div>", unsafe_allow_html=True)
