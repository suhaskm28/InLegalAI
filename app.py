import streamlit as st

# Logo and Title
st.set_page_config(page_title="LegalAI 🇮🇳", layout="wide")
st.sidebar.image("assets/logo.png", width=120)
st.sidebar.markdown("## 🧠 Features")

# Sidebar Links to Other Pages
st.sidebar.markdown("### Features")
st.sidebar.markdown("[🏠 Legal QA](./1_Legal_QA.py)")
st.sidebar.markdown("[📄 Document Analysis](./2_Document_Analysis.py)")
st.sidebar.markdown("[⚖️ Case Analysis](./3_Case_Analysis.py)")
st.sidebar.markdown("[📝 Complaint Drafting](./4_Complaint_Drafting.py)")

# Welcome Message
st.title("Welcome to LegalAI 🇮🇳")
st.write("Your personal legal assistant for Indian legal cases, document insights, and drafting support.")
st.info("Select a feature from the sidebar to begin.")
