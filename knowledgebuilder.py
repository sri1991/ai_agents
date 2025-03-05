import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, List, Tuple
from openai import AzureOpenAI
import os
import numpy as np
from numpy.linalg import norm

# Step 1: Define the Agent State
class AgentState(TypedDict):
    process_text_chunks: List[str]  # Chunks of process document text
    user_metadata: Dict            # Metadata provided by the user
    vector_store: FAISS            # Pre-existing vector store
    chunk_embeddings: np.ndarray   # Embeddings of process chunks
    mappings: List[Tuple[str, Document, float]]  # Mapped chunks with documents and scores

# Step 2: Load and Chunk Process Document
def load_and_chunk_process(state: AgentState) -> AgentState:
    loader = PyPDFLoader("process_doc.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = [chunk.page_content for chunk in text_splitter.split_documents(documents)]
    state["process_text_chunks"] = chunks
    print(f"Chunked process document into {len(chunks)} sections.")
    return state

# Step 3: Initialize Azure OpenAI Embedding Client
def initialize_embeddings(state: AgentState) -> AgentState:
    client = AzureOpenAI(
        azure_endpoint="YOUR_AZURE_ENDPOINT",
        api_key="YOUR_API_KEY",
        api_version="2023-05-15",  # Adjust based on your deployment
        azure_deployment="YOUR_EMBEDDING_DEPLOYMENT_NAME"  # e.g., "text-embedding-ada-002"
    )
    state["embedding_client"] = client
    print("Initialized Azure OpenAI embedding client.")
    return state

# Step 4: Generate Embeddings for Process Chunks
def generate_chunk_embeddings(state: AgentState) -> AgentState:
    client = state["embedding_client"]
    chunks = state["process_text_chunks"]
    
    # Generate embeddings
    response = client.embeddings.create(input=chunks, model="YOUR_EMBEDDING_DEPLOYMENT_NAME")
    embeddings = np.array([item.embedding for item in response.data])
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    state["chunk_embeddings"] = normalized_embeddings
    print(f"Generated embeddings for {len(chunks)} chunks.")
    return state

# Step 5: Load Pre-existing Vector Store (Assume from previous agent)
def load_vector_store(state: AgentState) -> AgentState:
    # Load or simulate the pre-existing vector store (replace with actual loading logic)
    # For this example, we'll assume it's passed or loaded from a file
    # Placeholder: Use the vector store from the previous agent
    from previous_agent import AgentState as PrevState, create_agent as prev_create_agent  # Hypothetical import
    prev_agent = prev_create_agent()
    prev_result = prev_agent.invoke({"excel_data": pd.read_excel("data.xlsx"), ...})  # Adjust initial state
    state["vector_store"] = prev_result["vector_store"]
    print("Loaded pre-existing vector store.")
    return state

# Step 6: Map Chunks to Vector Store with Metadata Filtering and Threshold
def map_with_metadata(state: AgentState) -> AgentState:
    vector_store = state["vector_store"]
    chunk_embeddings = state["chunk_embeddings"]
    user_metadata = state["user_metadata"]
    threshold = 0.85  # Similarity threshold
    
    # Get stored embeddings and documents from the vector store
    stored_embeddings = vector_store.index.reconstruct_n(0, vector_store.index.ntotal)
    norms = np.linalg.norm(stored_embeddings, axis=1, keepdims=True)
    normalized_stored_embeddings = stored_embeddings / norms
    documents = [vector_store.docstore.search(i) for i in range(vector_store.index.ntotal)]
    
    # Filter documents based on user metadata
    filtered_docs = [doc for doc in documents if all(doc.metadata.get(key) == value for key, value in user_metadata.items())]
    if not filtered_docs:
        print("No documents match the provided metadata.")
        state["mappings"] = []
        return state
    
    filtered_indices = [i for i, doc in enumerate(documents) if doc in filtered_docs]
    filtered_embeddings = normalized_stored_embeddings[filtered_indices]
    
    # Compute similarity scores for each chunk against filtered embeddings
    mappings = []
    for chunk, chunk_emb in zip(state["process_text_chunks"], chunk_embeddings):
        scores = np.dot(filtered_embeddings, chunk_emb)
        for idx, score in enumerate(scores):
            if score >= threshold:
                mappings.append((chunk, filtered_docs[idx], score))
    
    state["mappings"] = mappings
    print(f"Mapped {len(mappings)} chunks with similarity above threshold {threshold}.")
    return state

# Step 7: Build the Workflow
def create_agent():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("load_and_chunk_process", load_and_chunk_process)
    workflow.add_node("initialize_embeddings", initialize_embeddings)
    workflow.add_node("generate_chunk_embeddings", generate_chunk_embeddings)
    workflow.add_node("load_vector_store", load_vector_store)
    workflow.add_node("map_with_metadata", map_with_metadata)
    
    # Define edges
    workflow.add_edge("load_and_chunk_process", "initialize_embeddings")
    workflow.add_edge("initialize_embeddings", "generate_chunk_embeddings")
    workflow.add_edge("generate_chunk_embeddings", "load_vector_store")
    workflow.add_edge("load_vector_store", "map_with_metadata")
    workflow.add_edge("map_with_metadata", END)
    
    # Set entry point
    workflow.set_entry_point("load_and_chunk_process")
    
    return workflow.compile()

# Main Execution
def main():
    # User-provided metadata
    user_metadata = {"Category": "Security", "Status": "Active"}
    
    # Create and run the agent
    agent = create_agent()
    initial_state = {
        "process_text_chunks": [],
        "user_metadata": user_metadata,
        "vector_store": None,
        "chunk_embeddings": None,
        "mappings": []
    }
    
    result = agent.invoke(initial_state)
    
    # Display results
    if result["mappings"]:
        print("\nMapped Sections:")
        for chunk, doc, score in result["mappings"]:
            print(f"Process Chunk: {chunk}")
            print(f"Matched Document: {doc.page_content}, Score: {score:.4f}")
            print(f"Metadata: {doc.metadata}")
            print("---")
    else:
        print("No relevant mappings found.")

if __name__ == "__main__":
    os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_AZURE_ENDPOINT"
    os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_API_KEY"
    
    main()
