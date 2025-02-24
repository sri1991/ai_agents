from typing import List, Dict, Any, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
import logging
from pydantic import BaseModel
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGState(BaseModel):
    """State definition for the RAG pipeline"""
    messages: List[Any]
    context: List[Document] = []
    query: str = ""
    response: str = ""

class RAGAgent:
    """RAG Agent using LangGraph for document question answering"""
    
    def __init__(self, vectorstore: FAISS, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize the RAG Agent
        
        Args:
            vectorstore (FAISS): Initialized FAISS vector store
            model_name (str): Name of the LLM model to use
            temperature (float): Temperature for LLM responses
        """
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.graph = self._create_graph()
        
        # Define RAG prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context. 
            Use the context to provide accurate and relevant answers. If the context doesn't contain 
            enough information to answer the question, acknowledge this and suggest what additional 
            information might be needed.
            
            Context:
            {context}
            """),
            ("human", "{question}")
        ])

    def _retrieve(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents from vector store
        
        Args:
            state (RAGState): Current state
            
        Returns:
            RAGState: Updated state with retrieved documents
        """
        try:
            # Get relevant documents
            docs = self.vectorstore.similarity_search(state.query, k=3)
            state.context = docs
            logger.info(f"Retrieved {len(docs)} documents")
            return state
            
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return state

    def _generate_response(self, state: RAGState) -> RAGState:
        """
        Generate response using LLM
        
        Args:
            state (RAGState): Current state
            
        Returns:
            RAGState: Updated state with generated response
        """
        try:
            # Prepare context
            context_text = "\n\n".join([doc.page_content for doc in state.context])
            
            # Generate response
            chain = self.prompt | self.llm
            response = chain.invoke({
                "context": context_text,
                "question": state.query
            })
            
            state.response = response.content
            return state
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            state.response = f"Error generating response: {str(e)}"
            return state

    def _create_graph(self) -> Graph:
        """
        Create the LangGraph processing graph
        
        Returns:
            Graph: Configured processing graph
        """
        # Create workflow
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate_response)
        
        # Add edges
        workflow.add_edge('retrieve', 'generate')
        workflow.set_entry_point("retrieve")
        
        # Set final node
        workflow.set_finish_point("generate")
        
        return workflow.compile()

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline
        
        Args:
            question (str): User question
            
        Returns:
            Dict[str, Any]: Query results including response and context
        """
        try:
            # Initialize state
            initial_state = RAGState(
                messages=[HumanMessage(content=question)],
                query=question
            )
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            return {
                'query': question,
                'response': result.response,
                'context': [
                    {
                        'content': doc.page_content,
                        'source': doc.metadata.get('source', 'Unknown'),
                        'score': doc.metadata.get('score', None)
                    }
                    for doc in result.context
                ]
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'query': question,
                'error': str(e),
                'response': None,
                'context': []
            }

class StreamlitRAGApp:
    """Streamlit interface for RAG Agent"""
    
    def __init__(self, vectorstore: FAISS = None):
        """
        Initialize the Streamlit RAG application
        
        Args:
            vectorstore (FAISS): Optional pre-initialized vector store
        """
        self.setup_streamlit()
        self.vectorstore = vectorstore
        self.rag_agent = None
        
        if vectorstore:
            self.initialize_agent()
    
    def setup_streamlit(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="RAG Query Interface",
            page_icon="🤖",
            layout="wide"
        )
        
        st.title("🤖 RAG Query Interface")
        st.markdown("""
        Ask questions about your documents using RAG (Retrieval Augmented Generation).
        The system will:
        1. Find relevant content from your documents
        2. Generate an informed response using the context
        """)
    
    def initialize_agent(self):
        """Initialize the RAG agent"""
        if not self.vectorstore:
            st.warning("No vector store available. Please upload documents first.")
            return False
            
        try:
            self.rag_agent = RAGAgent(self.vectorstore)
            return True
        except Exception as e:
            st.error(f"Error initializing RAG agent: {str(e)}")
            return False
    
    def display_results(self, results: Dict[str, Any]):
        """
        Display query results
        
        Args:
            results (Dict[str, Any]): Query results
        """
        if 'error' in results:
            st.error(f"Error: {results['error']}")
            return
            
        # Display response
        st.subheader("Response")
        st.write(results['response'])
        
        # Display supporting context
        st.subheader("Supporting Context")
        for idx, context in enumerate(results['context'], 1):
            with st.expander(f"Context {idx} - Source: {context['source']}", expanded=False):
                st.write(context['content'])
                if context['score']:
                    st.info(f"Relevance Score: {context['score']:.4f}")
    
    def run(self):
        """Run the Streamlit application"""
        if not self.vectorstore:
            st.warning("Please upload and process documents first to create the knowledge base.")
            return
            
        if not self.rag_agent:
            if not self.initialize_agent():
                return
        
        # Query interface
        query = st.text_input(
            "Enter your question:",
            help="Ask a question about your documents"
        )
        
        if query:
            with st.spinner("Processing query..."):
                try:
                    results = self.rag_agent.query(query)
                    self.display_results(results)
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

def main():
    """Main entry point"""
    # This would typically be initialized with a vector store from document processing
    app = StreamlitRAGApp()
    app.run()

if __name__ == "__main__":
    main() 