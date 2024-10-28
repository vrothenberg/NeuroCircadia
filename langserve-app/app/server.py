#!/usr/bin/env python
# langserve-app/app/server.py
"""Example LangChain server exposes a retriever."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from app.chatbot import chatbot_chain, InputChat
# from app.rag_qa import rag_qa_chain, ChatHistory
import uuid
from literalai import LiteralClient
from dotenv import load_dotenv

load_dotenv()

client = LiteralClient()


def per_req_config_modifier(config, request):
    config["callbacks"] = [client.langchain_callback()]
    return config


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)


@app.middleware("http")
async def set_context_vars(request: Request, call_next):
    # Reset context vars
    client.reset_context()
    # Set thread_id
    thread_id = request.headers.get("thread_id") or uuid.uuid4().hex
    async with client.thread(thread_id=thread_id):
        response = await call_next(request)

    return response


# Adds routes to the app for using the retriever under:
# /invoke
# /batch
# /stream
add_routes(
    app,
    chatbot_chain.with_types(input_type=InputChat),
    path="/chatbot",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)


# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")
