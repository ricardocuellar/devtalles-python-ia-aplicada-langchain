
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


def index_texts(
    texts: list[str],
    metadatas: list[dict] = None,
    collection_name: str = "default",
    persist_path: str = None
) -> Chroma:
    """Indexa una lista de textos en ChromaDB"""
    path = persist_path or settings.CHROMA_PATH
    os.makedirs(path, exist_ok=True)

    documents = [
        Document(
            page_content=text,
            metadata=metadatas[index] if metadatas else {"index": index}

        )
        for index, text in enumerate(texts)
    ]

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        persist_directory=path,
        collection_name=collection_name
    )

    print(f"{len(documents)} documentos indexados en '{collection_name}'")

    return vectorstore


def index_pdf(
    pdf_path: str,
    collection_name: str = "pdf_collection",
    persist_path: str = None
) -> Chroma:
    """Carga un PDF, lo divide en chunks y lo indexa"""
    path = persist_path or settings.CHROMA_PATH
    os.makedirs(path, exist_ok=True)

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    print(f"{len(pages)} páginas cargadas de {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(pages)
    print(f"{len(chunks)} chunks creados")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=path,
        collection_name=collection_name
    )

    print(f"PDF indexado en '{collection_name}'")

    return vectorstore
