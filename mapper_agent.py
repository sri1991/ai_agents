# Import required libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

# Global variables for the knowledge base
knowledge_base = None

# Step 1: Ingest the standard policy document into a FAISS index
def ingest_standard_policy(policy_file: str = "standard_policy.pdf"):
    global knowledge_base
    # Load the standard policy document
    loader = PyPDFLoader(policy_file)
    documents = loader.load()

    # Split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Generate embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    chunk_texts = [chunk.page_content for chunk in chunks]
    embeddings = embedder.encode(chunk_texts, convert_to_numpy=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Store in global knowledge base
    knowledge_base = {
        "index": index,
        "chunks": chunk_texts,
        "embeddings": embeddings
    }
    print(f"Standard policy ingested with {len(chunks)} chunks.")

# Step 2: Define the agent's state and workflow
class AgentState(TypedDict):
    new_process_chunks: List[str]  # Chunks from the new process document
    matched_policy_chunks: List[Dict[str, float]]  # Matched policy chunks with similarity scores
    gaps: List[str]  # Identified gaps

# Load and chunk the new process document
def load_new_process(state: AgentState) -> AgentState:
    loader = PyPDFLoader("new_process.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    state["new_process_chunks"] = [chunk.page_content for chunk in chunks]
    return state

# Match new process chunks to policy using FAISS
def match_to_policy(state: AgentState) -> AgentState:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    new_embeddings = embedder.encode(state["new_process_chunks"], convert_to_numpy=True)
    distances, indices = knowledge_base["index"].search(new_embeddings, k=1)
    state["matched_policy_chunks"] = [
        {"text": knowledge_base["chunks"][idx[0]], "distance": dist[0]}
        for dist, idx in zip(distances, indices)
    ]
    return state

# Identify gaps based on similarity threshold
def identify_gaps(state: AgentState) -> AgentState:
    gaps = []
    threshold = 0.5  # Adjust this threshold as needed
    for i, (new_chunk, match) in enumerate(zip(state["new_process_chunks"], state["matched_policy_chunks"])):
        if match["distance"] > threshold:
            gaps.append(
                f"Gap in new process chunk {i+1}: '{new_chunk[:50]}...' "
                f"not covered by policy (closest match: '{match['text'][:50]}...', distance: {match['distance']:.2f})"
            )
    state["gaps"] = gaps
    return state

# Build and compile the workflow
def create_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("load_new_process", load_new_process)
    workflow.add_node("match_to_policy", match_to_policy)
    workflow.add_node("identify_gaps", identify_gaps)

    workflow.add_edge("load_new_process", "match_to_policy")
    workflow.add_edge("match_to_policy", "identify_gaps")
    workflow.add_edge("identify_gaps", END)

    workflow.set_entry_point("load_new_process")
    return workflow.compile()

# Main execution
def main():
    # Ingest the standard policy
    ingest_standard_policy("standard_policy.pdf")

    # Create the agent
    agent = create_agent()

    # Initial state
    initial_state = {
        "new_process_chunks": [],
        "matched_policy_chunks": [],
        "gaps": []
    }

    # Run the agent
    result = agent.invoke(initial_state)

    # Print results
    print("\nIdentified Gaps:")
    if result["gaps"]:
        for gap in result["gaps"]:
            print(gap)
    else:
        print("No significant gaps identified.")

if __name__ == "__main__":
    main()