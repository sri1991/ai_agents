import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # Replace with your LLM provider

# Step 1: Load Excel and Convert to Graph
def load_excel_to_graph(file_path: str = "data.xlsx"):
    # Read Excel file
    df = pd.read_excel(file_path)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges from the DataFrame
    for _, row in df.iterrows():
        source = row["Source"]
        target = row["Target"]
        relation = row["Relation"]
        details = row["Details"]
        
        # Add nodes (if not already added)
        G.add_node(source, type="entity")
        G.add_node(target, type="entity")
        
        # Add edge with relation and details
        G.add_edge(source, target, relation=relation, details=details)
    
    return G

# Step 2: Prepare Graph Data for Retrieval
def graph_to_documents(G: nx.DiGraph):
    # Convert graph edges to text documents for embedding
    documents = []
    for source, target, data in G.edges(data=True):
        text = f"{source} {data['relation']} {target}: {data['details']}"
        documents.append(Document(page_content=text, metadata={"source": source, "target": target, "relation": data["relation"]}))
    return documents

# Step 3: Build Vector Store
def build_vector_store(documents):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode([doc.page_content for doc in documents], convert_to_numpy=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Use LangChain FAISS wrapper
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, embedding) for doc, embedding in zip(documents, embeddings)],
        embedding=embedder
    )
    return vector_store

# Step 4: Query the Graph with RAG
def query_graph(G: nx.DiGraph, vector_store, query: str, llm):
    # Retrieve relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Top 3 matches
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
    
    # Enhance with graph traversal (optional)
    relevant_nodes = set()
    for doc in sources:
        relevant_nodes.add(doc.metadata["source"])
        relevant_nodes.add(doc.metadata["target"])
    
    # Fetch additional graph context
    graph_context = []
    for node in relevant_nodes:
        for neighbor, data in G[node].items():
            graph_context.append(f"{node} {data['relation']} {neighbor}: {data['details']}")
    
    # Combine RAG answer with graph context
    final_answer = f"{answer}\n\nAdditional Graph Context:\n" + "\n".join(graph_context)
    return final_answer

# Main Pipeline
def main():
    # Load Excel and create graph
    G = load_excel_to_graph("data.xlsx")
    print("Graph created with nodes:", G.nodes(), "and edges:", G.edges())
    
    # Convert graph to documents
    documents = graph_to_documents(G)
    
    # Build vector store
    vector_store = build_vector_store(documents)
    
    # Initialize LLM (replace with your preferred LLM, e.g., Grok via xAI API)
    llm = OpenAI(temperature=0.7)  # Placeholder; use xAI API if available
    
    # Example queries
    queries = [
        "What processes require safety checks?",
        "What is the reporting frequency in the process?"
    ]
    
    # Process queries
    for query in queries:
        print(f"\nQuery: {query}")
        answer = query_graph(G, vector_store, query, llm)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
