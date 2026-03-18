from langchain_core.tools import tool

@tool
def execute_sql(query: str) -> str:
    """
    Executes a SQL query against the database.
    (Currently a placeholder mock implementation)
    """
    return "SQL Execution is currently not configured. Please use dataframes for local analysis."
