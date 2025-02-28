import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
import os

# Step 1: Define the Agent State
class AgentState(TypedDict):
    risk_data: pd.DataFrame  # Excel risk definitions
    process_text: List[str]  # Chunks of process document text
    risk_mappings: List[Dict]  # Identified risks and mappings

# Step 2: Load Excel Risk Definitions
def load_risk_definitions(state: AgentState) -> AgentState:
    df = pd.read_excel("risk_definitions.xlsx")
    state["risk_data"] = df
    print(f"Loaded {len(df)} risk definitions from Excel.")
    return state

# Step 3: Load and Chunk Process Document
def load_process_document(state: AgentState) -> AgentState:
    loader = PyPDFLoader("process_doc.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    state["process_text"] = [chunk.page_content for chunk in text_splitter.split_documents(documents)]
    print(f"Loaded process document with {len(state['process_text'])} chunks.")
    return state

# Step 4: Identify and Map Risks
def identify_risks(state: AgentState) -> AgentState:
    # Initialize Azure OpenAI Chat
    llm = AzureOpenAI(
        azure_endpoint="YOUR_AZURE_ENDPOINT",
        api_key="YOUR_API_KEY",
        deployment_name="YOUR_DEPLOYMENT_NAME",
        api_version="2023-05-15",  # Adjust based on your Azure version
        temperature=0.7
    )
    
    # Prepare risk definitions as text
    risk_text = "\n".join(
        f"Risk ID: {row['Risk ID']} | Definition: {row['Risk Definition']} | Control: {row['Control Type']} | Mitigation: {row['Mitigation Strategy']}"
        for _, row in state["risk_data"].iterrows()
    )
    
    # Define prompt for LLM
    prompt_template = PromptTemplate(
        input_variables=["risks", "process"],
        template="""
        You are an AI advisor analyzing a process document against predefined risk definitions. Your task is to:
        1. Read the risk definitions below.
        2. Analyze the process document text.
        3. Identify risks in the process document and map them to the risk definitions.
        4. For each identified risk, provide:
           - The process section where the risk occurs
           - The matching Risk ID (or "None" if no match)
           - A brief explanation
           - A confidence score (0 to 1)

        **Risk Definitions**:
        {risks}

        **Process Document**:
        {process}

        Output your findings in this format:
        - Process Section: [section text]
          Risk ID: [id]
          Explanation: [reasoning]
          Confidence: [score]
        """
    )
    
    # Analyze each process chunk
    mappings = []
    for chunk in state["process_text"]:
        prompt = prompt_template.format(risks=risk_text, process=chunk)
        response = llm(prompt)
        mappings.append({"chunk": chunk, "mapping": response})
    
    state["risk_mappings"] = mappings
    return state

# Step 5: Build the Workflow
def create_agent():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("load_risk_definitions", load_risk_definitions)
    workflow.add_node("load_process_document", load_process_document)
    workflow.add_node("identify_risks", identify_risks)
    
    # Define edges
    workflow.add_edge("load_risk_definitions", "load_process_document")
    workflow.add_edge("load_process_document", "identify_risks")
    workflow.add_edge("identify_risks", END)
    
    # Set entry point
    workflow.set_entry_point("load_risk_definitions")
    
    return workflow.compile()

# Main Execution
def main():
    # Create and run the agent
    agent = create_agent()
    initial_state = {
        "risk_data": None,
        "process_text": [],
        "risk_mappings": []
    }
    
    result = agent.invoke(initial_state)
    
    # Print results
    print("\nRisk Mapping Results:")
    for mapping in result["risk_mappings"]:
        print(f"Process Chunk: {mapping['chunk'][:50]}...")
        print(f"Mapping:\n{mapping['mapping']}\n")

if __name__ == "__main__":
    # Set Azure OpenAI environment variables (optional, if not hardcoded)
    os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_AZURE_ENDPOINT"
    os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_API_KEY"
    
    main()
