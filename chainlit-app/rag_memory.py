# rag_memory.py

from operator import itemgetter
from typing import List, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from vertexai.language_models import TextEmbeddingModel
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
import chainlit as cl
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()

# Define prompt templates
RAG_TEMPLATE = """Given the following conversational history and user question, \
respond only with a search query to retrieve useful information from a medical \
knowledgebase. Do not narrate your thought process. If no relevant question is \
provided, respond only with "No valid query".

Chat History:
{chat_history}
Question: {question}
Search Query:"""

ANSWER_TEMPLATE = """You are NeuroCircadia, a scientific expert in human wellness, health \
optimization, and athletic performance. Help the user with relevant \
context from a medical knowledgebase if available. If the user's question is \
unrelated, politely reassert your role.

Chat History:
{chat_history}

Research Context:
{context}

Question: {question}"""

# Instantiate templates
RAG_QUERY_PROMPT = PromptTemplate.from_template(RAG_TEMPLATE)
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

# Helper function to combine retrieved documents
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# Setup embedding model and retriever
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
client = QdrantClient(url="http://localhost:6333")
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="uptodate",
    embedding=VertexAIEmbeddings(model_name="text-embedding-004")
)
retriever = vectorstore.as_retriever()

# User input schema
class ChatHistory(BaseModel):
    """Schema for chat history with the bot."""
    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str

# Main RAG-based conversational QA chain
def setup_runnable(user_session) -> None:
    """
    Sets up a Retrieval-Augmented Generation (RAG) pipeline to handle conversational 
    question-answering by retrieving relevant context from a knowledgebase.
    
    Args:
        user_session: The Chainlit user session object, which provides access to session 
                      data and allows setting the RAG runnable pipeline.

    Pipeline Components:
        1. Memory Retrieval: Accesses conversation history stored in memory dynamically.
        2. Query Generation: Uses chat history and user question to form a retrieval query.
        3. Document Retrieval: Queries the Qdrant vector store to obtain relevant documents.
        4. Answer Generation: Constructs the final response based on retrieved context.

    The final pipeline is stored in `user_session` and processes user messages with 
    enhanced contextual understanding.
    """
    memory = user_session.get("memory") 

    rprint(memory)

    # Step 1: Format inputs by retrieving history and question
    _inputs = RunnableMap(
        chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
        question=itemgetter("question"),
    )

    # # Step 2: Generate RAG query from formatted inputs
    # _rag_query = (
    #     _inputs
    #     | RAG_QUERY_PROMPT
    #     | ChatVertexAI(model="gemini-1.5-flash", temperature=0)
    #     | StrOutputParser()
    # )

    # Step 3: Retrieve documents based on RAG query and format context for answer generation
    _context = RunnableMap(
        context=(
            RAG_QUERY_PROMPT 
            | ChatVertexAI(model="gemini-1.5-flash", temperature=0) 
            | StrOutputParser() 
            | retriever 
            | _combine_documents),
        question=itemgetter("question"),
        chat_history=itemgetter("chat_history")
    )

    # Step 4: Complete RAG pipeline with answer generation
    rag_pipeline = (
        _inputs
        | _context
        | ANSWER_PROMPT
        | ChatVertexAI(model="gemini-1.5-flash", temperature=0)
        | StrOutputParser()
    ).with_types(input_type=ChatHistory)

    # Set the RAG pipeline in the user session
    user_session.set("runnable", rag_pipeline)
