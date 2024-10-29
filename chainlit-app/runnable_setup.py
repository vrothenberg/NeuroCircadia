# runnable_setup.py

from operator import itemgetter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_google_vertexai import ChatVertexAI
import chainlit as cl
from rich import print as rprint


def setup_runnable(user_session) -> None:
    """
    Sets up the main runnable pipeline with a conversation model, prompt, and memory history.

    Args:
        user_session: The Chainlit user session object, used to access memory and set the runnable.
    """
    memory = user_session.get("memory")  # Retrieve conversation memory
    rprint(memory)
    model = ChatVertexAI(model="gemini-1.5-flash", stream=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    # Define the runnable pipeline with memory, prompt, and model
    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )

    # Set runnable in user session
    user_session.set("runnable", runnable)
