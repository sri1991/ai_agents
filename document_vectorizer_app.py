import streamlit as st
from typing import List, Dict, Any
import logging
from pathlib import Path
import os
from datetime import datetime
import json
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentVectorizerApp:
    """Streamlit application for document processing and vector storage"""
    
    SUPPORTED_TYPES = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    
    def __init__(self):
        """Initialize the application components"""
        self.setup_streamlit()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize session state
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []

    def setup_streamlit(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Document Vectorizer",
            page_icon="📚",
            layout="wide"
        )
        
        st.title("📚 Document Vectorizer")
        st.markdown("""
        Upload documents to:
        1. Extract text content
        2. Create vector embeddings
        3. Store in FAISS vector database
        
        Supported formats: PDF, DOCX
        """)

    def process_document(self, file, file_path: str) -> Dict[str, Any]:
        """
        Process a single document
        
        Args:
            file: Uploaded file object
            file_path (str): Path to save the file temporarily
            
        Returns:
            Dict[str, Any]: Processing results
        """
        try:
            # Save uploaded file temporarily
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())
            
            # Load document based on type
            ext = Path(file.name).suffix.lower()
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            # Load and split document
            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)
            
            # Create or update vector store
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = FAISS.from_documents(
                    texts, 
                    self.embeddings
                )
            else:
                st.session_state.vectorstore.add_documents(texts)
            
            return {
                'filename': file.name,
                'chunks': len(texts),
                'total_chars': sum(len(t.page_content) for t in texts),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file.name}: {str(e)}")
            return {
                'filename': file.name,
                'error': str(e),
                'success': False
            }
        finally:
            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

    def process_files(self, files) -> Dict[str, Any]:
        """
        Process multiple uploaded files
        
        Args:
            files: List of uploaded files
            
        Returns:
            Dict[str, Any]: Processing results
        """
        results = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                ext = Path(file.name).suffix.lower()
                if ext not in self.SUPPORTED_TYPES:
                    results.append({
                        'filename': file.name,
                        'error': 'Unsupported file type',
                        'success': False
                    })
                    continue
                
                # Process file
                file_path = os.path.join(temp_dir, file.name)
                result = self.process_document(file, file_path)
                results.append(result)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'vectorstore_size': len(st.session_state.vectorstore.index) if st.session_state.vectorstore else 0
        }

    def display_results(self, results: Dict[str, Any]):
        """Display processing results"""
        st.subheader("Processing Results")
        
        # Display summary
        successful = sum(1 for r in results['results'] if r['success'])
        failed = len(results['results']) - successful
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Processed", len(results['results']))
        with col2:
            st.metric("Successful", successful)
        with col3:
            st.metric("Failed", failed)
        
        # Display vector store info
        st.info(f"Vector Store Size: {results['vectorstore_size']} vectors")
        
        # Display detailed results
        with st.expander("Detailed Results", expanded=True):
            for result in results['results']:
                if result['success']:
                    st.success(f"✅ {result['filename']}")
                    st.write(f"Chunks: {result['chunks']}")
                    st.write(f"Total Characters: {result['total_chars']}")
                else:
                    st.error(f"❌ {result['filename']}: {result['error']}")

    def save_to_history(self, results: Dict[str, Any]):
        """Save results to session history"""
        st.session_state.processing_history.append(results)

    def show_history(self):
        """Display processing history"""
        if st.session_state.processing_history:
            st.subheader("Processing History")
            
            for idx, entry in enumerate(reversed(st.session_state.processing_history)):
                with st.expander(f"Batch {idx + 1} - {entry['timestamp']}"):
                    self.display_results(entry)

    def search_documents(self, query: str, k: int = 5):
        """
        Search the vector store for similar documents
        
        Args:
            query (str): Search query
            k (int): Number of results to return
        """
        if not st.session_state.vectorstore:
            st.warning("No documents in the vector store. Please upload some documents first.")
            return
        
        try:
            results = st.session_state.vectorstore.similarity_search_with_score(query, k=k)
            
            st.subheader("Search Results")
            for doc, score in results:
                with st.expander(f"Score: {score:.4f}", expanded=True):
                    st.write(doc.page_content)
                    st.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        
        except Exception as e:
            st.error(f"Error during search: {str(e)}")

    def run(self):
        """Run the Streamlit application"""
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Select one or more PDF or DOCX files"
        )

        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    # Process files
                    results = self.process_files(uploaded_files)
                    
                    # Display results
                    self.display_results(results)
                    
                    # Save to history
                    self.save_to_history(results)
                    
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")

        # Search interface
        st.divider()
        st.subheader("Search Documents")
        query = st.text_input("Enter your search query:")
        k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        
        if query:
            self.search_documents(query, k)

        # Show processing history
        st.divider()
        self.show_history()

def main():
    """Main entry point"""
    app = DocumentVectorizerApp()
    app.run()

if __name__ == "__main__":
    main() 