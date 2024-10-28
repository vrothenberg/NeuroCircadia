# chatbot.py
from typing import List, Union
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI


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

chatbot_chain = (prompt | model).with_types(input_type=InputChat)
