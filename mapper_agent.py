# Import required libraries
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
import faiss
import numpy as np
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for the knowledge base
knowledge_base = None

class MapperAgent:
    """Agent for mapping and analyzing documents against standard policies"""
    
    def __init__(self):
        """Initialize the mapper agent"""
        # Initialize Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",  # Your Azure embedding deployment name
            openai_api_version="2024-02-15-preview"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

    def ingest_standard_policy(self, policy_file: str = "standard_policy.pdf") -> None:
        """
        Ingest the standard policy document into a FAISS index
        
        Args:
            policy_file (str): Path to the policy PDF file
        """
        global knowledge_base
        try:
            # Load the standard policy document
            loader = PyPDFLoader(policy_file)
            documents = loader.load()

            # Split into smaller chunks
            chunks = self.text_splitter.split_documents(documents)
            chunk_texts = [chunk.page_content for chunk in chunks]

            # Generate embeddings using Azure OpenAI
            embeddings_list = self.embeddings.embed_documents(chunk_texts)
            embeddings_array = np.array(embeddings_list)

            # Create FAISS index
            dimension = len(embeddings_list[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)

            # Store in global knowledge base
            knowledge_base = {
                "index": index,
                "chunks": chunk_texts,
                "embeddings": embeddings_array
            }
            logger.info(f"Standard policy ingested with {len(chunks)} chunks.")
            
        except Exception as e:
            logger.error(f"Error ingesting standard policy: {str(e)}")
            raise

# Step 2: Define the agent's state and workflow
class AgentState(TypedDict):
    new_process_chunks: List[str]  # Chunks from the new process document
    matched_policy_chunks: List[Dict[str, float]]  # Matched policy chunks with similarity scores
    gaps: List[str]  # Identified gaps

def load_new_process(state: AgentState) -> AgentState:
    """Load and chunk the new process document"""
    try:
        loader = PyPDFLoader("new_process.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        state["new_process_chunks"] = [chunk.page_content for chunk in chunks]
        return state
    except Exception as e:
        logger.error(f"Error loading new process: {str(e)}")
        raise

def match_to_policy(state: AgentState) -> AgentState:
    """Match new process chunks to policy using FAISS"""
    try:
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version="2024-02-15-preview"
        )
        
        # Generate embeddings for new chunks
        new_embeddings_list = embeddings.embed_documents(state["new_process_chunks"])
        new_embeddings = np.array(new_embeddings_list)
        
        # Search in FAISS index
        distances, indices = knowledge_base["index"].search(new_embeddings, k=1)
        
        state["matched_policy_chunks"] = [
            {"text": knowledge_base["chunks"][idx[0]], "distance": dist[0]}
            for dist, idx in zip(distances, indices)
        ]
        return state
        
    except Exception as e:
        logger.error(f"Error matching to policy: {str(e)}")
        raise

def identify_gaps(state: AgentState) -> AgentState:
    """Identify gaps based on similarity threshold"""
    try:
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
        
    except Exception as e:
        logger.error(f"Error identifying gaps: {str(e)}")
        raise

def create_agent():
    """Build and compile the workflow"""
    try:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("load_new_process", load_new_process)
        workflow.add_node("match_to_policy", match_to_policy)
        workflow.add_node("identify_gaps", identify_gaps)

        # Add edges
        workflow.add_edge("load_new_process", "match_to_policy")
        workflow.add_edge("match_to_policy", "identify_gaps")
        workflow.add_edge("identify_gaps", END)

        workflow.set_entry_point("load_new_process")
        return workflow.compile()
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise

def main():
    """Main execution"""
    try:
        # Initialize Azure OpenAI settings
        if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
            print("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables")
            return

        # Create mapper agent
        mapper = MapperAgent()
        
        # Ingest the standard policy
        mapper.ingest_standard_policy("standard_policy.pdf")

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
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()