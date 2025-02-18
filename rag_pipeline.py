from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document
import google.generativeai as genai
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    RAG Pipeline implementation using FAISS and Google's Gemini
    """
    
    def __init__(self, api_key: str, docs_dir: str = "documents"):
        """
        Initialize the RAG pipeline
        
        Args:
            api_key (str): Google API key
            docs_dir (str): Directory containing documents to process
        """
        self.api_key = api_key
        self.docs_dir = docs_dir
        
        # Configure Google AI
        genai.configure(api_key=api_key)
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.0-pro",
            temperature=0.7,
            google_api_key=api_key
        )
        
        # Initialize embeddings
        self.embeddings = GooglePalmEmbeddings(
            google_api_key=api_key
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Vector store
        self.vector_store = None

    def load_documents(self) -> List[Document]:
        """
        Load documents from the specified directory
        
        Returns:
            List[Document]: List of loaded documents
        """
        logger.info(f"Loading documents from {self.docs_dir}")
        
        try:
            # Create directory if it doesn't exist
            Path(self.docs_dir).mkdir(parents=True, exist_ok=True)
            
            # Load documents from directory
            loader = DirectoryLoader(
                self.docs_dir,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            return []

    def process_documents(self, documents: List[Document]) -> None:
        """
        Process documents and create vector store
        
        Args:
            documents (List[Document]): List of documents to process
        """
        logger.info("Processing documents")
        
        try:
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(texts)} chunks")
            
            # Create vector store
            self.vector_store = FAISS.from_documents(
                texts,
                self.embeddings
            )
            logger.info("Vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")

    def save_vector_store(self, path: str = "vector_store") -> None:
        """
        Save the vector store to disk
        
        Args:
            path (str): Path to save the vector store
        """
        if self.vector_store:
            self.vector_store.save_local(path)
            logger.info(f"Vector store saved to {path}")

    def load_vector_store(self, path: str = "vector_store") -> None:
        """
        Load the vector store from disk
        
        Args:
            path (str): Path to load the vector store from
        """
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(
                path,
                self.embeddings
            )
            logger.info(f"Vector store loaded from {path}")

    def query(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Query the RAG pipeline
        
        Args:
            query (str): Query string
            k (int): Number of relevant documents to retrieve
            
        Returns:
            Dict[str, Any]: Query results including relevant documents and generated response
        """
        if not self.vector_store:
            return {"error": "Vector store not initialized"}
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query, k=k)
            
            # Construct prompt with context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            prompt = f"""
            Based on the following context, please answer the question.
            If the answer cannot be derived from the context, say "I cannot answer this based on the provided context."

            Context:
            {context}

            Question: {query}
            """
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            return {
                "query": query,
                "relevant_documents": [doc.page_content for doc in relevant_docs],
                "response": response.content
            }
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return {"error": str(e)}

def main():
    """Example usage of the RAG pipeline"""
    # Replace with your actual API key
    api_key = "your_google_api_key_here"
    
    # Initialize pipeline
    pipeline = RAGPipeline(api_key)
    
    # Load and process documents
    documents = pipeline.load_documents()
    if documents:
        pipeline.process_documents(documents)
        
        # Save vector store
        pipeline.save_vector_store()
    
    # Example query
    query = "What are the main benefits of artificial intelligence in healthcare?"
    results = pipeline.query(query)
    
    # Print results
    print("\nQuery Results:")
    print(f"Query: {results['query']}")
    print("\nRelevant Documents:")
    for i, doc in enumerate(results.get('relevant_documents', []), 1):
        print(f"\nDocument {i}:")
        print(doc)
    print("\nGenerated Response:")
    print(results.get('response', 'No response generated'))

if __name__ == "__main__":
    main() 