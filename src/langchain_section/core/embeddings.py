
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchain_section.core.llm import get_embeddings
from src.langchain_section.config.settings import settings


def get_or_create_vectorstore(
    collection_name: str,
    persist_path: str = None
) -> Chroma:
    """Obtener un vector store existente"""
    path = persist_path or settings.CHROMA_PATH
    os.makedirs(path, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=path,
    )
