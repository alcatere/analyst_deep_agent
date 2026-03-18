from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

def get_python_agent_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are the Data Analysis Expert. Write python pandas code to answer questions about the user's dataframes."),
        ("user", "{input}")
    ])

def get_python_agent(model_name: str = "qwen3.5:9b"):
    """
    A specialized agent for Pandas / Python evaluation.
    """
    llm = ChatOllama(model=model_name, temperature=0.0)
    prompt = get_python_agent_prompt()
    return prompt | llm
