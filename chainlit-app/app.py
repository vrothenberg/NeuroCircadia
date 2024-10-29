# chainlit-app/app.py

from typing import Dict, Optional
from operator import itemgetter
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

import chainlit as cl
from chainlit.types import ThreadDict
from data_layer import SQLAlchemyDataLayer
from models import User

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_vertexai import ChatVertexAI
from langchain.schema.runnable.config import RunnableConfig 


# Database setup
engine = create_engine("sqlite:///chainlit_app.db")
SessionLocal = sessionmaker(bind=engine)
db_session = SessionLocal()



def setup_runnable():
    """Configure the Gemini Flash model with a prompt template and message history."""
    memory = ChatMessageHistory()
    model = ChatVertexAI(model="gemini-1.5-flash", stream=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="history"),  # 'history' expects a list of messages
        ("human", "{question}"),
    ])

    # Define the runnable pipeline
    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(lambda _: memory.messages)  # Ensure 'history' is a list of messages
        )
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)
    cl.user_session.set("memory", memory)



@cl.oauth_callback
def oauth_callback(provider_id: str, token: str, raw_user_data: Dict[str, str],
                   default_user: cl.User) -> Optional[cl.User]:
    """Callback after successful OAuth login."""
    # Return the default user authenticated through Google OAuth
    return default_user


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session with memory and runnable."""
    current_user = cl.user_session.get("user")  # Retrieve the authenticated user
    if current_user:
        cl.user_session.set("session", SQLAlchemyDataLayer(db_session))
        cl.user_session.set("thread_id", f"{current_user.identifier}_thread")
        setup_runnable()
        await cl.Message("Welcome! How can I assist you today?").send()
    else:
        await cl.Message("User not found. Please log in again.").send()



@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Resume a previous chat session with stored messages."""
    memory = ChatMessageHistory()
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]

    # Load past conversation into memory
    for message in root_messages:
        if message["type"] == "user_message":
            memory.add_user_message(message["output"])
        else:
            memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)
    setup_runnable()
    await cl.Message("Welcome back! How can I continue helping you?").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Process and respond to user messages."""
    memory = cl.user_session.get("memory")
    runnable = cl.user_session.get("runnable")

    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    await res.send()

    # Update memory with new messages
    memory.add_user_message(message.content)
    memory.add_ai_message(res.content)