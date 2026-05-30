
from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class RAGAgentSate(TypedDict):
    """Estados"""
    messages: Annotated[list, add_messages]
    question: str
    retrieved_docs: list[str]
    response: str
    need_retrieval: bool
    sources: list[dict]
