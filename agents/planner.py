from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

def get_planner_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are the primary Planner agent. Your job is to understand the user's intent and delegate or solve the request."),
        ("user", "{input}")
    ])

def get_planner_agent(model_name: str = "qwen3.5:9b"):
    llm = ChatOllama(model=model_name, temperature=0.1)
    prompt = get_planner_prompt()
    return prompt | llm
