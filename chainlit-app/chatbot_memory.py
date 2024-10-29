# chatbot_memory.py

from operator import itemgetter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_google_vertexai import ChatVertexAI
import chainlit as cl
from rich import print as rprint


def setup_runnable(user_session) -> None:
    """
    Initializes and configures the main runnable pipeline for Chainlit's conversation model.
    
    This function encapsulates the setup of the conversational AI pipeline using 
    the Chainlit session's memory and configures a prompt and response model. By isolating
    this setup process, the function adheres to modular design principles, allowing 
    for clean separation of session management and runnable configuration.

    Args:
        user_session: The Chainlit session object, which provides access to conversation 
                      memory and allows setting the runnable pipeline.

    Pipeline Components:
        1. Memory History: Accesses the conversation history from the session's memory, 
           enabling the model to incorporate chat context in responses.
        2. Prompt Template: Configures the chatbot's prompt template, setting up a system 
           message and placeholders to structure the conversation context and user queries.
        3. AI Model: Utilizes `ChatVertexAI` as the conversation model, set to stream responses.
        4. Output Parsing: Uses `StrOutputParser` to manage model output in a structured format.

    Returns:
        None. The runnable pipeline is set directly in the `user_session`.
    """
    memory = user_session.get("memory")  # Retrieve conversation memory
    # rprint(memory)  # Debugging output for memory inspection (can be removed in production)
    
    # Initialize the model and prompt for the pipeline
    model = ChatVertexAI(model="gemini-1.5-flash", stream=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    # Define the runnable pipeline: combines memory, prompt, model, and output parsing
    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )

    # Store the runnable pipeline in the session for use in message handling
    user_session.set("runnable", runnable)
