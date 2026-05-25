
from langchain_core.runnables import RunnableWithMessageHistory

from src.langchain_section.chains.base import build_assistant_chain
from src.langchain_section.memory.base import BaseMemoryBackend


def build_chatbot(backend: BaseMemoryBackend) -> RunnableWithMessageHistory:
    """Construyendo chatbot
    Args: 
        backend: Cualquie implementación BaseMemory Backend
    Returns:
        Chatbot: listo para invocar con session_id
    """
    chain = build_assistant_chain()

    return RunnableWithMessageHistory(
        chain,
        backend.get_history,
        input_messages_key="input",
        history_messages_key="history"
    )
