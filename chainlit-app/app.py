# app.py

from typing import Dict, Optional
import chainlit as cl
from chainlit.types import ThreadDict
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable.config import RunnableConfig 
from runnable_setup import setup_runnable


@cl.oauth_callback
def oauth_callback(provider_id: str, token: str, raw_user_data: Dict[str, str],
                   default_user: cl.User) -> Optional[cl.User]:
    """Callback after successful OAuth login, returning the authenticated user."""
    return default_user


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initializes a new chat session with conversation memory and runnable setup."""
    memory = ConversationBufferMemory(return_messages=True)
    cl.user_session.set("memory", memory)

    # Set up the runnable pipeline
    setup_runnable(cl.user_session)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    """Resumes a previous chat session by repopulating the memory from stored messages."""
    memory = ConversationBufferMemory(return_messages=True)

    # Load past conversation messages into memory
    for message in thread["steps"]:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        elif message["type"] == "assistant_message":
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)
    setup_runnable(cl.user_session)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handles incoming user messages, processes responses through the runnable pipeline, 
    and updates conversation memory."""
    memory = cl.user_session.get("memory")
    runnable = cl.user_session.get("runnable")

    response_message = cl.Message(content="")

    # Stream model response as chunks
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await response_message.stream_token(chunk)

    await response_message.send()

    # Update memory with the new conversation messages
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(response_message.content)
