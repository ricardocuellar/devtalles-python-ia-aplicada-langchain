
from pathlib import Path
from shutil import ExecError

from langchain_section.memory.base import BaseMemoryBackend
from langchain_section.memory.postgresql_memory import PostgreSQLMemoryBackend
from langchain_section.memory.sqlite_memory import SQLiteMemoryBackend
from src.langchain_section.core.document_loader import load_directory, split_documents
from src.langchain_section.core.embeddings import get_or_create_vectorstore

DOCUMENTS_DIR = Path("data/documents")
COLLECTION_NAME = "knwoledge_base"
CHROMA_PATH = "./data/chromadb_knwoledge"


def setup_vectorstore():
    """Carga o inicializa el vector store"""
    vectorstore = get_or_create_vectorstore(
        collection_name=COLLECTION_NAME,
        persist_path=CHROMA_PATH
    )

    count = vectorstore._collection.count()

    if count > 0:
        print(f" Base de conocimiento: {count} chunks indexados")
        return vectorstore

    print("Base de conocimiento vacía. Indexando documentos...")
    docs = load_directory(DOCUMENTS_DIR)

    if not docs:
        print(f"\n Agrega archivos .txt o .pdf en {DOCUMENTS_DIR}")
        return None

    chunks = split_documents(docs)
    vectorstore.add_documents(chunks)

    print(f"{len(chunks)} chunks indexados")
    return vectorstore


def setup_memory_backend() -> BaseMemoryBackend:
    """Intentar conectar a PostgreSQL o en su defecto usar SQLite"""
    try:
        backend = PostgreSQLMemoryBackend()
        backend.list_sessions()
        print("Memoria: PostgreSQL")
        return backend
    except Exception as e:
        print(f"❌ PostgreSQL no disponible {e}")
        print(" Usando SQLite como fallback de desarrollo")
        return SQLiteMemoryBackend()
