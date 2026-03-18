from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from .embeddings import get_embeddings

# Singleton pattern for the vector store
_vector_store = None

def get_vector_store(persist_directory: str = "./data/chroma_db"):
    global _vector_store
    
    if _vector_store is None:
        embeddings = get_embeddings()
        _vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return _vector_store

def add_documents_to_store(docs: List[Document], persist_directory: str = "./data/chroma_db"):
    """Adds a list of chunks to the ChromaDB vector store."""
    global _vector_store
    embeddings = get_embeddings()
    if _vector_store is None:
        _vector_store = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    else:
        _vector_store.add_documents(documents=docs)
    return True

def retrieve_similar(query: str, k: int = 3):
    store = get_vector_store()
    return store.similarity_search(query, k=k)
