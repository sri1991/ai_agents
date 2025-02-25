from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessMapperState(TypedDict):
    """State for the process mapper agent"""
    new_process_text: str
    knowledge_base_1: str
    knowledge_base_2: str
    gaps_identified: List[Dict[str, str]]
    current_step: str
    error: str | None

class ProcessMapperAgent:
    """Agent for mapping processes and identifying gaps"""
    
    def __init__(self, kb_folder: str = "knowledge_base"):
        """
        Initialize the Process Mapper Agent
        
        Args:
            kb_folder (str): Folder containing knowledge base documents
        """
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-35-turbo",
            openai_api_version="2024-02-15-preview",
            temperature=0.0
        )
        
        self.kb_folder = kb_folder
        self.gap_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a process analysis expert. Your task is to identify gaps between 
            a new process and existing knowledge bases. Compare the new process with both knowledge 
            bases and identify:
            1. Missing steps or requirements
            2. Inconsistencies in the process flow
            3. Additional controls or measures needed
            4. Any potential risks or compliance issues
            
            Knowledge Base 1:
            {knowledge_base_1}
            
            Knowledge Base 2:
            {knowledge_base_2}
            
            New Process:
            {new_process}
            
            Provide a detailed analysis of gaps found. Format your response as a list of gaps, 
            each with a 'category' and 'description'."""),
            ("human", "Please analyze the process and identify all gaps.")
        ])

    def load_knowledge_bases(self) -> tuple[str, str]:
        """
        Load knowledge base documents
        
        Returns:
            tuple[str, str]: Content of both knowledge bases
        """
        try:
            kb_files = list(Path(self.kb_folder).glob("*.pdf"))
            if len(kb_files) < 2:
                raise ValueError(f"Need at least 2 knowledge base documents in {self.kb_folder}")
            
            # Load first two knowledge base documents
            kb1_loader = PyPDFLoader(str(kb_files[0]))
            kb2_loader = PyPDFLoader(str(kb_files[1]))
            
            kb1_docs = kb1_loader.load()
            kb2_docs = kb2_loader.load()
            
            kb1_text = "\n".join(doc.page_content for doc in kb1_docs)
            kb2_text = "\n".join(doc.page_content for doc in kb2_docs)
            
            return kb1_text, kb2_text
            
        except Exception as e:
            logger.error(f"Error loading knowledge bases: {str(e)}")
            raise

    def read_process_document(self, state: ProcessMapperState) -> ProcessMapperState:
        """
        Read and process the new process document
        
        Args:
            state (ProcessMapperState): Current state
            
        Returns:
            ProcessMapperState: Updated state
        """
        try:
            state["current_step"] = "reading_process"
            
            # Load the process document
            loader = PyPDFLoader("new_process.pdf")
            documents = loader.load()
            
            # Combine all text
            state["new_process_text"] = "\n".join(doc.page_content for doc in documents)
            
            # Load knowledge bases
            kb1_text, kb2_text = self.load_knowledge_bases()
            state["knowledge_base_1"] = kb1_text
            state["knowledge_base_2"] = kb2_text
            
            logger.info("Successfully loaded process document and knowledge bases")
            return state
            
        except Exception as e:
            error_msg = f"Error reading process document: {str(e)}"
            logger.error(error_msg)
            state["error"] = error_msg
            return state

    def analyze_gaps(self, state: ProcessMapperState) -> ProcessMapperState:
        """
        Analyze gaps between process and knowledge bases
        
        Args:
            state (ProcessMapperState): Current state
            
        Returns:
            ProcessMapperState: Updated state
        """
        try:
            state["current_step"] = "analyzing_gaps"
            
            # Generate gap analysis using Azure OpenAI
            response = self.llm.invoke(
                self.gap_analysis_prompt.format(
                    knowledge_base_1=state["knowledge_base_1"],
                    knowledge_base_2=state["knowledge_base_2"],
                    new_process=state["new_process_text"]
                )
            )
            
            # Parse the response into structured gaps
            gaps = []
            current_gap = {}
            
            for line in response.content.split("\n"):
                line = line.strip()
                if line.startswith("Category:"):
                    if current_gap:
                        gaps.append(current_gap)
                    current_gap = {"category": line[9:].strip()}
                elif line.startswith("Description:"):
                    current_gap["description"] = line[12:].strip()
            
            if current_gap:
                gaps.append(current_gap)
            
            state["gaps_identified"] = gaps
            logger.info(f"Identified {len(gaps)} gaps in the process")
            return state
            
        except Exception as e:
            error_msg = f"Error analyzing gaps: {str(e)}"
            logger.error(error_msg)
            state["error"] = error_msg
            return state

    def create_workflow(self) -> StateGraph:
        """
        Create the agent workflow
        
        Returns:
            StateGraph: Configured workflow
        """
        # Create workflow
        workflow = StateGraph(ProcessMapperState)
        
        # Add nodes
        workflow.add_node("read_process", self.read_process_document)
        workflow.add_node("analyze_gaps", self.analyze_gaps)
        
        # Add edges
        workflow.add_edge("read_process", "analyze_gaps")
        workflow.add_edge("analyze_gaps", END)
        
        # Set entry point
        workflow.set_entry_point("read_process")
        
        return workflow.compile()

def main():
    """Main execution"""
    try:
        # Check for Azure OpenAI settings
        if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
            print("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables")
            return

        # Initialize agent
        agent = ProcessMapperAgent(kb_folder="knowledge_base")
        
        # Create workflow
        workflow = agent.create_workflow()
        
        # Initial state
        initial_state: ProcessMapperState = {
            "new_process_text": "",
            "knowledge_base_1": "",
            "knowledge_base_2": "",
            "gaps_identified": [],
            "current_step": "",
            "error": None
        }
        
        # Run the workflow
        result = workflow.invoke(initial_state)
        
        # Check for errors
        if result["error"]:
            print(f"Error: {result['error']}")
            return
        
        # Print results
        print("\nGap Analysis Results:")
        print("-" * 50)
        
        for i, gap in enumerate(result["gaps_identified"], 1):
            print(f"\nGap {i}:")
            print(f"Category: {gap['category']}")
            print(f"Description: {gap['description']}")
        
        if not result["gaps_identified"]:
            print("No gaps identified in the process.")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 