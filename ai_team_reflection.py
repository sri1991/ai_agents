from typing import List, Dict, Any
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import logging
from datetime import datetime
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReflectionLog:
    """Data class to store reflection results"""
    timestamp: str
    agent: str
    task: str
    outcome: str
    improvements: List[str]
    confidence_score: float

class SelfReflectiveAgent:
    """Base class for agents with self-reflection capabilities"""
    
    def __init__(self, name: str, role: str, goal: str, api_key: str):
        genai.configure(api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.0-pro",
            temperature=0.7,
            google_api_key=api_key
        )
        
        self.agent = Agent(
            name=name,
            role=role,
            goal=goal,
            backstory=f"An AI agent specialized in {role} with self-reflection capabilities",
            llm=self.llm,
            verbose=True
        )
        
        self.reflection_logs: List[ReflectionLog] = []

    def reflect_on_outcome(self, task: str, outcome: str) -> ReflectionLog:
        """
        Perform self-reflection on task outcome
        
        Args:
            task (str): The task that was performed
            outcome (str): The outcome of the task
            
        Returns:
            ReflectionLog: Reflection analysis results
        """
        reflection_prompt = f"""
        Analyze the following task and its outcome:
        Task: {task}
        Outcome: {outcome}
        
        Please provide:
        1. List of potential improvements
        2. Confidence score (0-1) for the outcome
        3. Specific actionable feedback
        
        Format your response as JSON with keys: 'improvements', 'confidence_score', 'feedback'
        """
        
        try:
            response = self.llm.invoke(reflection_prompt)
            reflection_data = json.loads(response.content)
            
            log = ReflectionLog(
                timestamp=datetime.now().isoformat(),
                agent=self.agent.name,
                task=task,
                outcome=outcome,
                improvements=reflection_data['improvements'],
                confidence_score=float(reflection_data['confidence_score'])
            )
            
            self.reflection_logs.append(log)
            return log
            
        except Exception as e:
            logger.error(f"Reflection failed: {str(e)}")
            return None

class ResearchTeam:
    """Manages a team of self-reflective AI agents"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Initialize specialized agents
        self.researcher = SelfReflectiveAgent(
            name="Research Analyst",
            role="research and data analysis",
            goal="Conduct thorough research and provide data-driven insights",
            api_key=api_key
        )
        
        self.writer = SelfReflectiveAgent(
            name="Content Writer",
            role="content creation and synthesis",
            goal="Create clear, engaging content from research findings",
            api_key=api_key
        )
        
        self.reviewer = SelfReflectiveAgent(
            name="Quality Reviewer",
            role="quality assurance and validation",
            goal="Ensure accuracy and quality of deliverables",
            api_key=api_key
        )

    def create_research_crew(self, research_topic: str) -> Crew:
        """
        Create a crew for research tasks with self-reflection capabilities
        
        Args:
            research_topic (str): The topic to research
            
        Returns:
            Crew: Configured crew for the research task
        """
        # Define tasks with reflection points
        research_task = Task(
            description=f"Research and analyze {research_topic}. Include key findings and data points.",
            agent=self.researcher.agent,
            expected_output="Comprehensive research findings with supporting data",
            context="Ensure thorough coverage of the topic with credible sources"
        )
        
        writing_task = Task(
            description="Create a detailed report based on the research findings",
            agent=self.writer.agent,
            expected_output="Well-structured report with clear sections and insights",
            context="Focus on clarity and engagement while maintaining accuracy"
        )
        
        review_task = Task(
            description="Review and validate the report for accuracy and quality",
            agent=self.reviewer.agent,
            expected_output="Validation report with specific feedback",
            context="Check for accuracy, clarity, and completeness"
        )
        
        # Create and return the crew
        return Crew(
            agents=[self.researcher.agent, self.writer.agent, self.reviewer.agent],
            tasks=[research_task, writing_task, review_task],
            process=Process.sequential,
            verbose=2
        )

    async def execute_research_project(self, topic: str) -> Dict[str, Any]:
        """
        Execute a complete research project with self-reflection
        
        Args:
            topic (str): Research topic
            
        Returns:
            Dict[str, Any]: Project results and reflection logs
        """
        crew = self.create_research_crew(topic)
        results = await crew.kickoff()
        
        # Perform self-reflection for each agent
        reflections = []
        for agent in [self.researcher, self.writer, self.reviewer]:
            reflection = agent.reflect_on_outcome(
                task=f"Complete {agent.agent.role} tasks for {topic}",
                outcome=results.get(agent.agent.name, "Task completed")
            )
            reflections.append(reflection)
        
        return {
            "results": results,
            "reflections": reflections
        }

async def main():
    """Example usage of the self-reflective research team"""
    api_key = "your_google_api_key_here"
    
    # Initialize research team
    team = ResearchTeam(api_key)
    
    # Execute research project
    research_topic = "Impact of Artificial Intelligence on Healthcare in 2024"
    project_results = await team.execute_research_project(research_topic)
    
    # Print results and reflections
    print("\nProject Results:")
    print(json.dumps(project_results["results"], indent=2))
    
    print("\nAgent Reflections:")
    for reflection in project_results["reflections"]:
        print(f"\nAgent: {reflection.agent}")
        print(f"Confidence Score: {reflection.confidence_score}")
        print("Improvements:")
        for imp in reflection.improvements:
            print(f"- {imp}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 