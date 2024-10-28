# question_answer.py
from operator import itemgetter
from typing import List, Tuple

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from vertexai.language_models import TextEmbeddingModel
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

RAG_TEMPLATE = """Given the following conversational history and user question, \
respond only with a search query to retrieve useful information from a medical \
knowledgebase. Do not narrate your thought process. If no relevant question is \
provided, respond only with "No valid query".

Chat History:
{chat_history}
Question: {question}
Search Query:"""

RAG_QUERY_PROMPT = PromptTemplate.from_template(RAG_TEMPLATE)

ANSWER_TEMPLATE = """You are a scientific expert in human wellness, health \
optimization, and athletic performance. Answer the user's question with relevant \
context from a medical knowledgebase if available. If the user's question is \
unrelated, politely reassert your role.

Context:
{context}

Question: {question}"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    """Format chat history into a string."""
    return "\n".join(
        [f"Human: {turn[0]}\nAssistant: {turn[1]}" for turn in chat_history]
        )


embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
client = QdrantClient(url="http://localhost:6333")
vectorstore = QdrantVectorStore(
    client=client,
    collection_name="uptodate",
    embedding=VertexAIEmbeddings(model_name="text-embedding-004")
)
retriever = vectorstore.as_retriever()

_inputs = RunnableMap(
    chat_history=lambda x: _format_chat_history(x["chat_history"]),
    question=lambda x: x["question"],
    rag_query=(
        RunnablePassthrough()
        | RAG_QUERY_PROMPT
        | ChatVertexAI(model="gemini-1.5-flash", temperature=0)
        | StrOutputParser()
    ),
)

_context = RunnableMap(
    context=itemgetter("rag_query") | retriever | _combine_documents,
    question=itemgetter("question"),
)

# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str


conversational_qa_chain = (
    _inputs
    | _context
    | ANSWER_PROMPT
    | ChatVertexAI(model="gemini-1.5-flash", temperature=0)
    | StrOutputParser()
)
rag_qa_chain = conversational_qa_chain.with_types(input_type=ChatHistory)