from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf(file_path: str) -> List[Document]:
    """Loads a PDF from the given file path and returns LangChain Documents."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs
