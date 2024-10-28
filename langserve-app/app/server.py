#!/usr/bin/env python
# langserve-app/app/server.py
"""Example LangChain server exposes a retriever."""
from fastapi import FastAPI, Request
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langserve import add_routes

from langchain_core.prompts import MessagesPlaceholder

from pydantic import BaseModel, Field
from typing import List, Union

import uuid

from literalai import LiteralClient
from dotenv import load_dotenv

load_dotenv()

client = LiteralClient()


class InputChat(BaseModel):
    """Input schema for the chatbot endpoint."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


model = ChatVertexAI(model="gemini-1.5-flash", stream=True)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
# runnable = prompt | model | StrOutputParser()

chatbot_chain = (prompt | model).with_types(input_type=InputChat)


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
    chatbot_chain,
    config_keys=["callbacks"],
    per_req_config_modifier=per_req_config_modifier,
    path="/test",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001, log_level="debug")