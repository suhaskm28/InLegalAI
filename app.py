import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"  # Fix PyTorch path bug

import streamlit as st

# Set Streamlit page config
st.set_page_config(page_title="LegalAI 🇮🇳", page_icon="⚖️", layout="wide")

# ---------- Inject Custom CSS ----------
st.markdown("""
    <style>
    /* Hide Streamlit's default main menu & footer */
    #MainMenu, footer {visibility: hidden;}

    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1f3b4d;
        margin-top: 1.2rem;
    }

    .subtitle {
        font-size: 1.3rem;
        color: #444;
        margin-bottom: 2rem;
    }

    .feature-button {
        border: none;
        width: 100%;
        text-align: left;
        background-color: #f1f3f5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 1rem;
        transition: 0.2s all ease-in-out;
        font-size: 1.1rem;
        font-weight: 500;
    }

    .feature-button:hover {
        background-color: #dbe7f0;
        transform: scale(1.01);
        cursor: pointer;
    }

    .logo-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 1rem;
    }

    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.image("assets/logo.png", width=140, output_format="auto", caption="LegalAI", use_column_width=False)
    st.markdown("### 🧠 Features")

    # Using buttons to simulate navigation
    if st.button("🏠 Legal QA"):
        st.switch_page("pages/1_Legal_QA.py")
    if st.button("📄 Document Analysis"):
        st.switch_page("pages/2_Document_Analysis.py")
    if st.button("⚖️ Case Analysis"):
        st.switch_page("pages/3_Case_Analysis.py")
    if st.button("📝 Complaint Drafting"):
        st.switch_page("pages/4_Complaint_Drafting.py")

# ---------- Main Content ----------
st.markdown("<div class='main-title'>Welcome to LegalAI 🇮🇳</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Your personal AI-powered assistant for Indian legal queries, case analysis, document insights, and legal drafting.</div>", unsafe_allow_html=True)

# ---------- Feature Grid ----------
col1, col2 = st.columns(2)

with col1:
    if st.button("💬 Legal QA", key="qa"):
        st.switch_page("pages/1_Legal_QA.py")
    st.markdown("""
        <div class="feature-button">
            Ask legal questions in plain English and get contextual answers grounded in Indian case law using our Retrieval-Augmented Generation (RAG) pipeline.
        </div>
    """, unsafe_allow_html=True)

    if st.button("📘 Document Analysis", key="doc"):
        st.switch_page("pages/2_Document_Analysis.py")
    st.markdown("""
        <div class="feature-button">
            Upload judgments, agreements or notices to get detailed clause-wise and section-wise legal insights.
        </div>
    """, unsafe_allow_html=True)

with col2:
    if st.button("📚 Case Analysis", key="case"):
        st.switch_page("pages/3_Case_Analysis.py")
    st.markdown("""
        <div class="feature-button">
            Dive deep into individual cases: explore court reasoning, precedents, citations and ask questions within the case context.
        </div>
    """, unsafe_allow_html=True)

    if st.button("✍️ Complaint Drafting", key="draft"):
        st.switch_page("pages/4_Complaint_Drafting.py")
    st.markdown("""
        <div class="feature-button">
            Automatically generate legal complaint drafts for civil and criminal cases using smart templates + AI understanding.
        </div>
    """, unsafe_allow_html=True)
