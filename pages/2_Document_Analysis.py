import streamlit as st

st.set_page_config(page_title="Document Analysis", layout="wide")
st.title("📄 Document Analysis")

st.write("Upload legal documents for semantic understanding and querying.")

uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])

if uploaded_file:
    st.success("File uploaded! (Processing logic goes here)")
