from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun

# Import our custom tools
from tools.retriever_tool import read_documents
from tools.python_tool import analyze_dataframe, generate_chart
from tools.sql_tool import execute_sql

from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Searches the internet for current information about the query."""
    search = DuckDuckGoSearchRun()
    try:
        return search.invoke(query)
    except Exception as e:
        return f"Error searching the web: {e}"

def create_workflow(model_name: str = "qwen3.5:9b"):
    """
    Constructs the main routing graph.
    Currently uses LangGraph's prebuilt ReAct agent for simplicity, 
    but the architecture allows for replacing with a custom StateGraph.
    """
    llm = ChatOllama(model=model_name, temperature=0.2)
    
    # Define available tools dynamically
    tools = [search_web, read_documents, analyze_dataframe, generate_chart, execute_sql]
    
    # Return the agent executor
    agent_executor = create_react_agent(llm, tools)
    return agent_executor
