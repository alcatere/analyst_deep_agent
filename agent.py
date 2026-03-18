import os
from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import pandas as pd

# Global state for vectors and dataframes (in a real app, use a proper DB)
vector_store = None
uploaded_dataframes = {}

# --- Tools ---

@tool
def search_web(query: str) -> str:
    """Searches the internet for current information about the query."""
    search = DuckDuckGoSearchRun()
    try:
        return search.invoke(query)
    except Exception as e:
        return f"Error searching the web: {e}"

@tool
def read_documents(query: str) -> str:
    """Searches previously uploaded PDF documents for the given query."""
    global vector_store
    if vector_store is None:
        return "No documents have been uploaded or indexed yet."
    
    docs = vector_store.similarity_search(query, k=3)
    if not docs:
        return "No relevant information found in the documents."
    
    context = "\n\n".join([f"Excerpt: {doc.page_content}" for doc in docs])
    return f"Found the following information in documents:\n{context}"

@tool
def analyze_dataframe(query: str, filename: str) -> str:
    """
    Analyzes an uploaded CSV/Dataframe.
    You must provide the user's question as 'query' and the 'filename'.
    """
    global uploaded_dataframes
    if filename not in uploaded_dataframes:
        return f"File '{filename}' not found. Available files: {list(uploaded_dataframes.keys())}"
    
    df = uploaded_dataframes[filename]
    
    # Very basic execution for analysis - in a production app use pandas-agent or LLM code execution safely.
    # For now, we will return df info and ask the LLM to reason about it, or use a basic python repl.
    # To keep it safe and simple, we just return the schema and head, and let the LLM figure it out,
    # or implement a safer subset. Let's return some stats.
    
    summary = f"Columns: {list(df.columns)}\nShape: {df.shape}\n"
    summary += f"Head:\n{df.head(3).to_string()}\n"
    summary += f"Description:\n{df.describe().to_string()}"
    return f"Data summary for {filename}:\n{summary}\nNote: I can only provide summary stats right now."

tools = [search_web, read_documents, analyze_dataframe]

# --- Agent Setup ---

def get_agent(model_name: str = "qwen2.5"):
    # Initialize the local LLM via Ollama
    # Note: the user must have Ollama running locally with the given model.
    llm = ChatOllama(model=model_name, temperature=0.2)
    
    # Create the ReAct agent using LangGraph
    agent_executor = create_react_agent(llm, tools)
    return agent_executor

# --- Helper Functions for File Uploads ---

def ingest_pdf(file_path: str):
    global vector_store
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text") # Requires nomic-embed-text or another embedding model
    
    if vector_store is None:
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    else:
        vector_store.add_documents(documents=splits)
    
    return f"Successfully indexed {len(splits)} chunks from the PDF."

def ingest_csv(file_path: str, filename: str):
    global uploaded_dataframes
    df = pd.read_csv(file_path)
    uploaded_dataframes[filename] = df
    return f"Successfully loaded CSV '{filename}' with {df.shape[0]} rows."
