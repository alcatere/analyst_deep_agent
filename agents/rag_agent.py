from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

def get_rag_agent_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are the Document Expert. Answer the user's question using ONLY the retrieved context."),
        ("user", "{input}")
    ])

def get_rag_agent(model_name: str = "qwen3.5:9b"):
    """
    A specialized agent that handles RAG tasks.
    In a full LangGraph, this would be a node taking context and summarizing it.
    """
    llm = ChatOllama(model=model_name, temperature=0.1)
    prompt = get_rag_agent_prompt()
    return prompt | llm
