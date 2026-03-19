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

import matplotlib.pyplot as plt

@tool
def generate_chart(filename: str, x_column: str, y_column: str, kind: str, title: str) -> str:
    """
    Generates a chart from the dataframe to visualize data trends.
    'kind' must be 'line', 'bar', 'scatter', or 'hist'.
    Use this tool whenever the user asks for graphics, plots, or visual charts of the database/CSV!
    """
    global uploaded_dataframes
    if filename not in uploaded_dataframes:
        return f"File '{filename}' not found."
    
    df = uploaded_dataframes[filename]
    try:
        plt.figure(figsize=(8, 5))
        if kind == 'line':
            plt.plot(df[x_column], df[y_column])
        elif kind == 'bar':
            plt.bar(df[x_column], df[y_column])
        elif kind == 'scatter':
            plt.scatter(df[x_column], df[y_column])
        elif kind == 'hist':
            plt.hist(df[x_column])
        else:
            return "Error: Unsupported chart kind."
            
        plt.title(title)
        plt.xlabel(x_column)
        plt.ylabel(y_column if kind != 'hist' else 'Frequency')
        plt.tight_layout()
        plt.savefig('temp_chart.png')
        plt.close()
        
        return "SUCCESS: Chart saved to temp_chart.png. Tell the user you have generated the graphic."
    except Exception as e:
        return f"Error generating chart: {e}"
