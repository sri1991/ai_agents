import streamlit as st
from pypdf2 import PdfReader
import pandas as pd
import os
import time

# Set page configuration
st.set_page_config(page_title="Swoosh Ecosystem", layout="centered", initial_sidebar_state="auto")

# Custom CSS to match the screenshot's look and feel
st.markdown(
    """
    <style>
    .header {
        background: linear-gradient(90deg, #003087, #0055a4); /* Gradient blue for branding */
        color: white;
        padding: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .header span:nth-child(2) {
        font-size: 20px;
        font-weight: normal;
    }
    .upload-section {
        padding: 25px;
        background-color: #f0f4f8;
        border: 1px solid #ddd;
        border-radius: 8px;
        text-align: center;
        max-width: 600px;
        margin: 0 auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stFileUploader {
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #28a745; /* Green from screenshot */
        color: white;
        border: none;
        padding: 10px 30px;
        border-radius: 20px;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #218838;
        transform: translateY(-2px);
    }
    .checkbox-label {
        font-size: 16px;
        margin: 10px 0;
        display: block;
    }
    .status {
        text-align: center;
        color: #555;
        margin-top: 20px;
    }
    .sidebar .sidebar-content {
        padding: 10px;
    }
    .sidebar .sidebar-content .stSelectbox label {
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with branding
st.markdown(
    '<div class="header">Standard Chartered<br><span>Swoosh Ecosystem</span></div>',
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = ["Raise IT Issue or Re...", "AI Agent Setup", "Doc Analyzer", "Inference"]
page = st.sidebar.selectbox("Go to", pages, index=2)  # Default to Doc Analyzer

# Main content - Doc Analyzer
if page == "Doc Analyzer":
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.write("Choose a PDF/TIFF/IMAGE/TXT/DOC File")
    
    # File uploader with enhanced UX
    uploaded_file = st.file_uploader("Select Files", type=["pdf", "txt", "doc", "tiff", "jpg", "png"], accept_multiple_files=False, key="file_uploader")
    
    # Checkboxes with labels
    extract = st.checkbox("Extract", value=False, key="extract_checkbox")
    signature = st.checkbox("Signature Detection", value=False, key="signature_checkbox")
    
    # Upload and Extract button with loading state
    if st.button("Upload and Extract", key="upload_button"):
        if uploaded_file is not None:
            with st.spinner("Processing file..."):
                time.sleep(2)  # Simulate processing
                file_type = uploaded_file.type
                if file_type == "application/pdf":
                    pdf_reader = PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""
                    st.success("File processed successfully!")
                    st.write("Extracted Text (first 1000 chars):")
                    st.text(text[:1000])
                else:
                    st.error("File type not supported for extraction yet. Supported: PDF")
        
            if extract:
                st.write("Extraction process completed.")
            if signature:
                st.write("Signature detection process completed (placeholder).")
        else:
            st.warning("Please upload a file first!")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="status">Status: Ready</div>', unsafe_allow_html=True)

# Placeholder for other pages
elif page == "Raise IT Issue or Re...":
    st.write("Raise IT Issue or Request content goes here.")
elif page == "AI Agent Setup":
    st.write("AI Agent Setup content goes here.")
elif page == "Inference":
    st.write("Inference content goes here.")
