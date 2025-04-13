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
    # Display the logo
    st.image("assets/logo.png", width=140, caption="LegalAI", use_container_width=True)
    
    # Add a title for the features section
    st.markdown("### 🧠 Features")

    # Using buttons to simulate navigation
    if st.button("🏠 Legal QA"):
        st.session_state.current_page = "1_Legal_QA"  # Use session_state to track the page
        st.experimental_rerun()  # Rerun the app to switch pages

    if st.button("📄 Document Analysis"):
        st.session_state.current_page = "2_Document_Analysis"
        st.experimental_rerun()

    if st.button("⚖️ Case Analysis"):
        st.session_state.current_page = "3_Case_Analysis"
        st.experimental_rerun()

    if st.button("📝 Complaint Drafting"):
        st.session_state.current_page = "4_Complaint_Drafting"
        st.experimental_rerun()
# Add logic in the main app (app.py) to switch between pages based on session_state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "1_Legal_QA"  # Default page

# Switch to the selected page based on session_state
if st.session_state.current_page == "1_Legal_QA":
    # Render the page content for 1_Legal_QA.py
    st.write("### Legal QA Page")  # Replace with the actual content
elif st.session_state.current_page == "2_Document_Analysis":
    # Render the page content for 2_Document_Analysis.py
    st.write("### Document Analysis Page")  # Replace with the actual content
elif st.session_state.current_page == "3_Case_Analysis":
    # Render the page content for 3_Case_Analysis.py
    st.write("### Case Analysis Page")  # Replace with the actual content
elif st.session_state.current_page == "4_Complaint_Drafting":
    # Render the page content for 4_Complaint_Drafting.py
    st.write("### Complaint Drafting Page")  # Replace with the actual content

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
