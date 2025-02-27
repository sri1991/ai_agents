import pandas as pd
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.llms import OpenAI  # Replace with your LLM provider

# Step 1: Load Excel and Group Data
def load_excel_and_group(file_path: str = "input.xlsx", group_by_column: str = "Category"):
    # Read Excel file
    df = pd.read_excel(file_path)
    print(f"Loaded Excel with {len(df)} rows.")
    
    # Group by the specified column
    grouped = df.groupby(group_by_column)
    
    # Convert grouped data into documents
    documents = []
    for group_name, group_df in grouped:
        # Combine content within the group
        combined_content = "\n".join(
            f"Section: {row['Section']} | Content: {row['Content']}"
            for _, row in group_df.iterrows()
        )
        text = f"Category: {group_name}\n{combined_content}"
        metadata = {"category": group_name, "sections": list(group_df["Section"])}
        documents.append(Document(page_content=text, metadata=metadata))
    
    print(f"Grouped into {len(documents)} documents.")
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
    source_info = "\n".join([f"Source: {doc.page_content[:100]}..." for doc in sources])
    final_answer = f"Answer: {answer}\n\nRetrieved Sources:\n{source_info}"
    return final_answer

# Main Pipeline
def main():
    # Load Excel and group data
    documents = load_excel_and_group("input.xlsx", group_by_column="Category")
    
    # Build vector store
    vector_store = build_vector_store(documents)
    
    # Initialize LLM (replace with your preferred LLM, e.g., Grok via xAI API)
    llm = OpenAI(temperature=0.7)  # Placeholder; use xAI API if available
    
    # Example queries
    queries = [
        "What safety-related information is there?",
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
