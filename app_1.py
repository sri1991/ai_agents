import streamlit as st
from pypdf2 import PdfReader
import pandas as pd
import os

# Set page configuration for a clean look
st.set_page_config(page_title="Swoosh Ecosystem", layout="centered", initial_sidebar_state="auto")

# Custom CSS to match the branding and layout
st.markdown(
    """
    <style>
    .header {
        background-color: #003087; /* Standard Chartered blue */
        color: white;
        padding: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .upload-section {
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 5px;
        text-align: center;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #004d99;
    }
    .checkbox-label {
        font-size: 16px;
        margin: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with branding
st.markdown('<div class="header">Standard Chartered<br>Swoosh Ecosystem</div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["AI Agent Setup", "Doc Analyzer", "Inference"]
page = st.sidebar.selectbox("Go to", pages, index=1)  # Default to Doc Analyzer

# Main content - Doc Analyzer
if page == "Doc Analyzer":
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.write("Choose a PDF/TIFF/IMAGE/TXT/DOC File")
    
    # File uploader
    uploaded_file = st.file_uploader("Select Files", type=["pdf", "txt", "doc", "tiff", "jpg", "png"], accept_multiple_files=False)
    
    # Checkboxes
    extract = st.checkbox("Extract", value=False)
    signature = st.checkbox("Signature Detection", value=False)
    
    # Upload and Extract button
    if st.button("Upload and Extract"):
        if uploaded_file is not None:
            # Process the file based on type
            file_type = uploaded_file.type
            if file_type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                st.write("Extracted Text:")
                st.text(text[:1000])  # Limit to 1000 chars for display
            else:
                st.write("File type not supported for extraction yet. Supported: PDF")
        
        if extract:
            st.write("Extraction process initiated...")
        if signature:
            st.write("Signature detection process initiated...")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Placeholder for other pages
elif page == "AI Agent Setup":
    st.write("AI Agent Setup content goes here.")
elif page == "Inference":
    st.write("Inference content goes here.")

# Footer or additional info (optional)
st.sidebar.write("© 2025 Standard Chartered")
