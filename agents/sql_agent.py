from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

def get_sql_agent_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are the SQL Database Expert. Your goal is to translate natural language into correct SQL queries."),
        ("user", "{input}")
    ])

def get_sql_agent(model_name: str = "qwen3.5:9b"):
    """
    A specialized agent for SQL operations.
    """
    llm = ChatOllama(model=model_name, temperature=0.0)
    prompt = get_sql_agent_prompt()
    return prompt | llm
