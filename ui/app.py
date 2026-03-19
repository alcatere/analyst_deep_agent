import streamlit as st
import os
import tempfile
from langchain_core.messages import HumanMessage, AIMessage

# Import from our new architecture
from graph.workflow import create_workflow
from ingestion.pdf_loader import load_pdf
from ingestion.chunking import split_documents
from rag.retriever import add_documents_to_store
from tools.python_tool import add_dataframe
import pandas as pd

st.set_page_config(page_title="Smart Analyst Agent", page_icon="🤖", layout="wide")

st.title("🤖 Smart Analyst Agent")
st.markdown("Powered by local models (Qwen via Ollama)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = create_workflow(model_name="qwen3.5:9b") 

with st.sidebar:
    st.header("⚙️ Configuration")
    model_name = st.text_input("Local Model Name (Ollama)", value="qwen3.5:9b")
    
    if st.button("Update Model"):
        st.session_state.agent = create_workflow(model_name=model_name)
        st.success(f"Agent updated to use {model_name}")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.success("Conversation cleared!")
        st.rerun()

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    st.header("📂 Upload Knowledge")
    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_pdf is not None and uploaded_pdf.name not in st.session_state.processed_files:
        with st.spinner("Indexing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.getvalue())
                tmp_path = tmp.name
            
            try:
                docs = load_pdf(tmp_path)
                chunks = split_documents(docs)
                add_documents_to_store(chunks)
                st.session_state.processed_files.add(uploaded_pdf.name)
                st.success(f"Successfully indexed {len(chunks)} chunks from '{uploaded_pdf.name}'.")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
            finally:
                os.unlink(tmp_path)

    uploaded_csv = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded_csv is not None and uploaded_csv.name not in st.session_state.processed_files:
        with st.spinner("Loading Dataset..."):
            try:
                df = pd.read_csv(uploaded_csv)
                add_dataframe(uploaded_csv.name, df)
                st.session_state.processed_files.add(uploaded_csv.name)
                st.success(f"Successfully loaded CSV '{uploaded_csv.name}' with {df.shape[0]} rows.")
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

from evaluation.metrics import evaluate_relevance
import shutil

# Chat interface
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)
        if "eval_score" in message.additional_kwargs:
            st.caption(f"🤖 Auto-Eval Score: {message.additional_kwargs['eval_score']}/5")
        if "image_path" in message.additional_kwargs:
            st.image(message.additional_kwargs["image_path"])

if prompt := st.chat_input("Ask me anything, or tell me to read your documents/data..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking and using tools..."):
            try:
                final_response = ""
                for chunk in st.session_state.agent.stream({"messages": st.session_state.messages}):
                    if "agent" in chunk:
                        ai_msg = chunk["agent"]["messages"][0]
                        final_response = ai_msg.content
                        
                    elif "tools" in chunk:
                        tool_msg = chunk["tools"]["messages"][0]
                        st.info(f"🛠️ Tool Used: {tool_msg.name}")
                
                message_placeholder.markdown(final_response)
                
                # Setup kwargs for this message
                kwargs = {}
                
                # Check for graphics
                if os.path.exists("temp_chart.png"):
                    # Save permanently so it persists in chat history
                    save_path = f"data/chart_{len(st.session_state.messages)}.png"
                    os.makedirs("data", exist_ok=True)
                    shutil.copy("temp_chart.png", save_path)
                    kwargs["image_path"] = save_path
                    os.remove("temp_chart.png")
                    st.image(save_path)
                
                # Evaluate and show score
                with st.spinner("Evaluating response..."):
                    score = evaluate_relevance(prompt, final_response, model_name=st.session_state.get("model_name", "qwen3.5:9b"))
                    kwargs["eval_score"] = score
                    st.caption(f"🤖 Auto-Eval Score: {score}/5")

                msg = AIMessage(content=final_response)
                msg.additional_kwargs = kwargs
                st.session_state.messages.append(msg)
                
            except Exception as e:
                st.error(f"Error communicating with agent. Make sure ollama is running. Error: {e}")
