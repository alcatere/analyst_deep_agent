import streamlit as st
import os
import tempfile
from agent import get_agent, ingest_pdf, ingest_csv
from langchain_core.messages import HumanMessage, AIMessage

# Page config
st.set_page_config(page_title="Smart Analyst Agent", page_icon="🤖", layout="wide")

st.title("🤖 Smart Analyst Agent")
st.markdown("Powered by local models (Qwen via Ollama)")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = get_agent(model_name="qwen2.5") # Adjust default if needed

# Sidebar for controls and file uploads
with st.sidebar:
    st.header("⚙️ Configuration")
    model_name = st.text_input("Local Model Name (Ollama)", value="qwen2.5")
    if st.button("Update Model"):
        st.session_state.agent = get_agent(model_name=model_name)
        st.success(f"Agent updated to use {model_name}")

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.success("Conversation cleared!")
        st.rerun()

    st.header("📂 Upload Knowledge")
    uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_pdf is not None:
        if st.button("Process PDF"):
            with st.spinner("Indexing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.getvalue())
                    tmp_path = tmp.name
                
                try:
                    result = ingest_pdf(tmp_path)
                    st.success(result)
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    os.unlink(tmp_path)

    uploaded_csv = st.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded_csv is not None:
        if st.button("Process CSV"):
            with st.spinner("Loading Dataset..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_csv.getvalue())
                    tmp_path = tmp.name
                
                try:
                    result = ingest_csv(tmp_path, uploaded_csv.name)
                    st.success(result)
                except Exception as e:
                    st.error(f"Error processing CSV: {e}")
                finally:
                    os.unlink(tmp_path)

# Chat interface
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

if prompt := st.chat_input("Ask me anything, or tell me to read your documents/data..."):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking and using tools..."):
            try:
                # The agent stream yields dictionaries containing updates from nodes
                # We want to pick the final agent message or stream the tool calls
                final_response = ""
                for chunk in st.session_state.agent.stream({"messages": st.session_state.messages}):
                    if "agent" in chunk:
                        ai_msg = chunk["agent"]["messages"][0]
                        final_response = ai_msg.content
                        
                    elif "tools" in chunk:
                        tool_msg = chunk["tools"]["messages"][0]
                        st.info(f"🛠️ Tool Used: Result -> {tool_msg.name}")
                
                message_placeholder.markdown(final_response)
                st.session_state.messages.append(AIMessage(content=final_response))
            except Exception as e:
                st.error(f"Error communicating with agent: {e}")
                st.info("Make sure 'ollama serve' is running and you have downloaded the correct models.")
