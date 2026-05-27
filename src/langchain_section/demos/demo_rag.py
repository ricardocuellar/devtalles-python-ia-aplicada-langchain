

from pathlib import Path

from src.langchain_section.core.embeddings import get_or_create_vectorstore
from src.langchain_section.core.document_loader import load_directory, split_documents


DOCUMENTS_DIR = Path("data/documents")
COLLECTION_NAME = "demo_rag_real"
CHROMA_PATH = "./data/chromadb_rag_demo"


def index_documents() -> tuple:
    """Carga archivos reales desde dicso y divide en chunks"""
    docs = load_directory(DOCUMENTS_DIR)

    if not docs:
        return None, 0  # (vectorstore, num_chunks)

    chunks = split_documents(docs)

    if not chunks:
        print("❌ No se generaron chunks. Los archivos podrían estar vacíos.")
        return None, 0

    vectorstore = get_or_create_vectorstore(
        collection_name=COLLECTION_NAME,
        persist_path=CHROMA_PATH
    )

    current_count = vectorstore._collection.count()

    if current_count > 0:
        print("\n. Ya hay {current_count} chunks indexados")
        answer = input("¿Reindexar desde cero? (s/N): ").strip().lower()

        if answer == "s":
            vectorstore._client.delete_collection(COLLECTION_NAME)
            vectorstore = get_or_create_vectorstore(
                collection_name=COLLECTION_NAME,
                persist_path=CHROMA_PATH
            )

            vectorstore.add_documents(chunks)
            print(f"Reindexado: {len(chunks)} chunks en ChromaDB")

        else:
            print(f"Usando índice existente ({current_count} chunks)")

    else:
        vectorstore.add_documents(chunks)
        print(f"{len(chunks)} chunks indexados en ChromaDB")

    return vectorstore, vectorstore._collection.count()


def show_used_sources(docs: list) -> None:
    """Muestra archivos y fragmentos que usó el sistema"""
    if not docs:
        return

    print("\nFuentes consultadas: ")

    viewed_sources = {}

    for doc in docs:
        name = doc.metadata.get("file_name", "desconocida")
        page = doc.metadata.get("page", None)
        start = doc.metadata.get("start_index", None)

        if name not in viewed_sources:
            viewed_sources[name] = []

        info = ""

        if page is not None:
            info = f"pág. {page+1}"
        elif start is not None:
            info = f"pos. {start}"

        if info:
            viewed_sources[name].append(info)

    for file, locations in viewed_sources.items():
        if locations:
            print(f" {file} ({', '.join(locations)})")
        else:
            print(f" {file}")
