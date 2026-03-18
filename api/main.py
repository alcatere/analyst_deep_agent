from fastapi import FastAPI
from pydantic import BaseModel
from graph.workflow import create_workflow

app = FastAPI(title="Smart Analyst API")

class ChatRequest(BaseModel):
    message: str
    model: str = "qwen3.5:9b"

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    """
    API endpoint for communicating with the agent framework.
    """
    # Note: In production, instantiate this once or persist state across sessions
    workflow = create_workflow(model_name=req.model)
    
    # A single turn chat for the API
    result = workflow.invoke({"messages": [("user", req.message)]})
    ai_msg = result["messages"][-1].content
    return {"response": ai_msg}
