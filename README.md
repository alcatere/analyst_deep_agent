# Smart Analyst Agent

A local, AI-powered smart analyst application built with Python, LangGraph, and Streamlit. This agent uses local Large Language Models (LLMs) via Ollama, ensuring that your data stays private and secure on your own machine. 

## Features

*   **Local AI Engine**: Uses `Qwen` models (or any other model supported by Ollama).
*   **Document Analysis (RAG)**: Upload PDFs. The app splits and embeds them into a local ChromaDB instance, allowing the agent to answer questions based entirely on your documents.
*   **Data Consulting**: Upload CSV files. The agent automatically analyzes the structure and summary statistics via Pandas, answering fundamental data-related questions.
*   **Web Search Capability**: Integrated DuckDuckGo search lets the agent fetch real-time information to augment its answers.
*   **Step-by-Step Reasoning**: Leverages the ReAct paradigm (Reasoning + Acting) via LangGraph, allowing the AI to logically plan and iterate until a complete answer is derived.

## Prerequisites

1.  **Python 3.10+**
2.  **Ollama**: Install [Ollama](https://ollama.com/) to run models locally.

## Setup Instructions

1.  **Install dependencies**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Download the LLM**:
    Make sure Ollama is running, then pull your preferred model (default is `qwen3.5:9b`):
    ```bash
    ollama pull qwen3.5:9b
    ```

3.  **Run the Application**:
    ```bash
    bash run.sh
    ```
    *(Alternatively, run `source venv/bin/activate && streamlit run app.py`)*

4.  **Usage**:
    *   The app will open in your browser (`http://localhost:8501`).
    *   Use the sidebar to upload PDFs or CSVs for the agent context.
    *   Chat with your Smart Analyst!

## Technologies Used

*   **UI Framework**: Streamlit
*   **Agent Framework**: LangChain & LangGraph
*   **Local LLMs**: Ollama integration
*   **Vector Store**: ChromaDB
*   **Data Manipulation**: Pandas
*   **Data Visualization**: Matplotlib & Seaborn
