# chainlit-app/app.py

from typing import Optional
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
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory
from langchain_google_vertexai import ChatVertexAI


# Initialize SQLite database connection and session
engine = create_engine("sqlite:///chainlit_app.db")
SessionLocal = sessionmaker(bind=engine)
db_session = SessionLocal()


def setup_runnable():
    """Set up the Gemini Flash model with a prompt template and memory."""
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model = ChatVertexAI(model="gemini-1.5-flash", streaming=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful chatbot"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Authenticate user with the database credentials."""
    user_record = db_session.query(User).filter_by(identifier=username).first()
    if user_record and user_record.password == password:
        user = cl.User(identifier=username)
        
        # Set up session data and thread ID
        session = SQLAlchemyDataLayer(db_session)
        cl.user_session.set("session", session)
        cl.user_session.set("thread_id", f"{user.identifier}_thread")
        
        return user
    return None


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new chat session with memory and runnable setup."""
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable()
    await cl.Message("Welcome! How can I assist you today?").send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Resume an existing chat session and load previous messages into memory."""
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [msg for msg in thread["steps"] if msg["parentId"] is None]
    
    # Load past conversation into memory
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)
    setup_runnable()
    await cl.Message("Welcome back! How can I continue helping you?").send()


@cl.on_message
async def on_message(message: cl.Message):
    """Process user messages and update memory and responses dynamically."""
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    runnable = cl.user_session.get("runnable")  # type: Runnable
    response_message = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await response_message.stream_token(chunk)

    await response_message.send()

    # Update memory with the new conversation
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(response_message.content)
