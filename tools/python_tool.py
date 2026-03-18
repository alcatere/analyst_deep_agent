import pandas as pd
from langchain_core.tools import tool

# Global state mimicking a database or active session dataframes
uploaded_dataframes = {}

def add_dataframe(filename: str, df: pd.DataFrame):
    global uploaded_dataframes
    uploaded_dataframes[filename] = df

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
    
    # We return basic stats, allowing the agent to reason over them.
    summary = f"Columns: {list(df.columns)}\nShape: {df.shape}\n"
    summary += f"Head:\n{df.head(3).to_string()}\n"
    summary += f"Description:\n{df.describe().to_string()}"
    return f"Data summary for {filename}:\n{summary}\nNote: I can only provide summary stats right now."
