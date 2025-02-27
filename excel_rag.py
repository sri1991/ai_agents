import pandas as pd
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.llms import OpenAI  # Replace with your LLM provider

# Step 1: Load Excel and Convert to Documents
def load_excel_to_documents(file_path: str = "input.xlsx"):
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Convert each row to a text document
    documents = []
    for _, row in df.iterrows():
        # Combine columns into a single string (customize as needed)
        text = f"Section: {row['Section']} | Content: {row['Content']} | Category: {row['Category']}"
        metadata = {"section": row["Section"], "category": row["Category"]}
        documents.append(Document(page_content=text, metadata=metadata))
    
    return documents

# Step 2: Build Vector Store
def build_vector_store(documents):
    # Initialize embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create FAISS vector store from documents
    vector_store = FAISS.from_documents(documents, embedder)
    return vector_store

# Step 3: Query the Data with RAG
def query_rag(vector_store, query: str, llm):
    # Set up retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 matches
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Run the query
    result = qa_chain({"query": query})
    answer = result["result"]
    sources = result["source_documents"]
    
    # Format output with source details
    source_info = "\n".join([f"Source: {doc.page_content}" for doc in sources])
    final_answer = f"Answer: {answer}\n\nRetrieved Sources:\n{source_info}"
    return final_answer

# Main Pipeline
def main():
    # Load Excel and convert to documents
    documents = load_excel_to_documents("input.xlsx")
    print(f"Loaded {len(documents)} documents from Excel.")
    
    # Build vector store
    vector_store = build_vector_store(documents)
    
    # Initialize LLM (replace with your preferred LLM, e.g., Grok via xAI API)
    llm = OpenAI(temperature=0.7)  # Placeholder; use xAI API if available
    
    # Example queries
    queries = [
        "What sections mention safety?",
        "What is the reporting frequency?",
        "Are there any finance-related processes?"
    ]
    
    # Process queries
    for query in queries:
        print(f"\nQuery: {query}")
        answer = query_rag(vector_store, query, llm)
        print(answer)

if __name__ == "__main__":
    main()
