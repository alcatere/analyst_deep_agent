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
import seaborn as sns

# Set a professional seaborn theme
sns.set_theme(style="whitegrid", palette="muted")

@tool
def generate_chart(filename: str, x_column: str, y_column: str, kind: str, title: str) -> str:
    """
    Generates a beautiful Seaborn chart to visualize data trends.
    'kind' must be 'line', 'bar', 'scatter', 'hist', 'boxplot', or 'violin'.
    Use this tool whenever the user asks for graphics, plots, or visual charts of the CSV!
    """
    global uploaded_dataframes
    if filename not in uploaded_dataframes:
        return f"File '{filename}' not found."
    
    df = uploaded_dataframes[filename]
    try:
        plt.figure(figsize=(10, 6))
        
        if kind == 'line':
            sns.lineplot(data=df, x=x_column, y=y_column)
        elif kind == 'bar':
            sns.barplot(data=df, x=x_column, y=y_column)
        elif kind == 'scatter':
            sns.scatterplot(data=df, x=x_column, y=y_column)
        elif kind == 'hist':
            sns.histplot(data=df, x=x_column, kde=True)
        elif kind == 'boxplot':
            sns.boxplot(data=df, x=x_column, y=y_column)
        elif kind == 'violin':
            sns.violinplot(data=df, x=x_column, y=y_column)
        else:
            return "Error: Unsupported chart kind."
            
        plt.title(title, fontsize=14, pad=15)
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column if kind != 'hist' else 'Frequency', fontsize=12)
        plt.tight_layout()
        plt.savefig('temp_chart.png', dpi=300)
        plt.close()
        
        # Calculate a quick statistical context to help the LLM write a better narrative
        context = f"Chart '{title}' saved to temp_chart.png successfully.\n"
        context += "Here is the statistical snapshot of the plotted columns to help you write your analysis narrative:\n"
        if df[x_column].dtype in ['int64', 'float64']:
            context += f"- {x_column} stats: Mean={df[x_column].mean():.2f}, Max={df[x_column].max():.2f}, Min={df[x_column].min():.2f}\n"
        if kind != 'hist' and df[y_column].dtype in ['int64', 'float64']:
            context += f"- {y_column} stats: Mean={df[y_column].mean():.2f}, Max={df[y_column].max():.2f}, Min={df[y_column].min():.2f}\n"
            
        # Try to calculate correlation if both are numeric
        if kind != 'hist' and df[x_column].dtype in ['int64', 'float64'] and df[y_column].dtype in ['int64', 'float64']:
            corr = df[x_column].corr(df[y_column])
            context += f"- Correlation between {x_column} and {y_column}: {corr:.2f}\n"

        return context
    except Exception as e:
        return f"Error generating chart: {e}"
