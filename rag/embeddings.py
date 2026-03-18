from langchain_ollama import OllamaEmbeddings

def get_embeddings(model_name: str = "nomic-embed-text") -> OllamaEmbeddings:
    """Returns local embeddings using Ollama."""
    return OllamaEmbeddings(model=model_name)
