# langserve-app/app/chatbot.py

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import List, Union, Dict, Any
from pydantic import BaseModel, Field

class InputChat(BaseModel):
    """Input schema for the chatbot endpoint."""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ..., description="Current conversation messages."
    )

model = ChatVertexAI(model="gemini-1.5-flash", stream=False)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're a knowledgeable historian providing accurate and eloquent answers to historical questions."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

def trim_chat_messages(messages: List[Union[HumanMessage, AIMessage, SystemMessage]]):
    """Trim messages to prevent context overflow."""
    return trim_messages(messages, max_tokens=30)

def format_messages(input: Dict[str, Any]) -> Dict[str, Any]:
    """Format messages for the prompt."""
    return {
        "history": input["messages"][:-1],
        "input": input["messages"][-1].content
    }

# Create the chain as a Runnable
chatbot_chain = (
    RunnablePassthrough()
    | RunnableLambda(format_messages)
    | prompt
    | model
)