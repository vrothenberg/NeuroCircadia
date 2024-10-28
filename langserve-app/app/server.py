#!/usr/bin/env python
# langserve-app/app/server.py

from fastapi import FastAPI, Request, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from app.chatbot import chatbot_chain, InputChat
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from typing import Optional, Dict, Any
from literalai import LiteralClient
import uuid
import os
from pydantic import BaseModel

from dotenv import load_dotenv
import contextvars

load_dotenv()
client = LiteralClient()

# Define memory persistence
memory = MemorySaver()

# User authentication mock database
users_db = {"admin": "password123"}

def authenticate_user(username: str, password: str) -> Optional[str]:
    return username if users_db.get(username) == password else None

class GraphState(BaseModel):
    chat_input: InputChat
    metadata: Dict[str, Any] = {}

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="LangChain API Server with stateful memory",
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define LangGraph workflow
workflow = StateGraph(GraphState)

def chatbot_state(state: GraphState) -> GraphState:
    response = chatbot_chain.invoke(state.chat_input.dict())
    state.metadata["last_response"] = response
    return state

# Add nodes and edges to the LangGraph workflow
workflow.add_node("chatbot", chatbot_state)
workflow.add_edge(START, "chatbot")

# Compile workflow with memory checkpointing
app_with_memory = workflow.compile(checkpointer=memory)

@app.middleware("http")
async def set_context_vars(request: Request, call_next):
    client.reset_context()
    thread_id = request.headers.get("thread_id") or str(uuid.uuid4())
    async with client.thread(thread_id=thread_id):
        response = await call_next(request)
    return response

@app.post("/login")
async def login(
    username: str = Body(..., embed=True),
    password: str = Body(..., embed=True)
):
    if authenticate_user(username, password):
        return {"username": username, "thread_id": str(uuid.uuid4())}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

# Add routes for the chatbot chain
add_routes(
    app,
    chatbot_chain,  # Now this is a proper Runnable
    path="/chatbot",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001, log_level="debug")