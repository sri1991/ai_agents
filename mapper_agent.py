import streamlit as st
from langchain.document_loaders import PyPDFLoader  # Or other loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict

# ------------------ 1. Define the State ------------------
class GraphState(TypedDict):
    document_text: str
    embeddings: List[List[float]]
    identified_gaps: List[str]
    report: str

# ------------------ 2. Define the Agents (Nodes) ------------------
def document_reader(document) -> str:
    # Load and read the document
    loader = PyPDFLoader(document) #Example for PDF
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    document_text = "".join([doc.page_content for doc in texts])
    return {"document_text": document_text}

def gap_identifier(state: GraphState) -> Dict:
    # Compare document embeddings to FAISS vector store
    # Identify gaps
    # Return identified gaps
    # Access the vectorstore
    index = FAISS.load_local("faiss_index", embeddings) #load your Faiss index
    standards = index.similarity_search(state["document_text"], k=3) #check vectorstore for similar documents
    #Process the standards and identify the gaps
    gaps = "These are the gaps from the document" #modify this code
    return {"identified_gaps": gaps}

def report_generator(state: GraphState) -> Dict:
    # Format the identified gaps into a report
    report = f"Report: {state['identified_gaps']}"
    return {"report": report}

# ------------------ 3. Define the Graph ------------------
def define_graph():
    builder = StateGraph(GraphState)
    builder.add_node("document_reader", document_reader)
    builder.add_node("gap_identifier", gap_identifier)
    builder.add_node("report_generator", report_generator)

    builder.set_entry_point("document_reader")
    builder.add_edge("document_reader", "gap_identifier")
    builder.add_edge("gap_identifier", "report_generator")
    builder.add_edge("report_generator", END)

    return builder.compile()

# ------------------ 4. Streamlit App ------------------
st.title("Process Gap Analyzer")
embeddings = OpenAIEmbeddings() # your embeddings
uploaded_file = st.file_uploader("Upload Process Document", type=["pdf", "txt", "docx"])

if uploaded_file is not None:
    graph = define_graph()
    inputs = {"document": uploaded_file}
    st.write("Running Graph")
    result = graph.invoke(inputs)
    st.write("Gaps Identified:")
    st.write(result['report']) #display the report
