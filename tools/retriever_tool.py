from langchain_core.tools import tool
from rag.retriever import retrieve_similar

@tool
def read_documents(query: str) -> str:
    """Searches previously uploaded PDF documents for the given query."""
    try:
        docs = retrieve_similar(query, k=3)
        if not docs:
            return "No relevant information found in the local documents."
        
        context = "\n\n".join([f"Excerpt: {doc.page_content}" for doc in docs])
        return f"Found the following information in documents:\n{context}"
    except Exception as e:
        return f"Warning: Vector store might not be initialized yet. Error: {str(e)}"
