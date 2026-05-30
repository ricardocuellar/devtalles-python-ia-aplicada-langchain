
import json


from langchain.messages import HumanMessage
from langchain_chroma import Chroma

from src.langchain_section.config.settings import settings
from src.langchain_section.core.llm import get_llm
from src.langchain_section.graphs.states import RAGAgentState


def node_analyze(state: RAGAgentState) -> dict:
    """Nodo que decide si la pregunta necesita buscar documentos"""
    llm = get_llm(temperature=0.1)

    recent_context = ""
    if state.get("messages") and len(state["messages"]) >= 2:
        last_two = state["messages"][-2:]
        recent_context = "\n".join([
            f"{'Usuario' if message.type == 'human' else 'IA'}: {message.content[:150]}"
            for message in last_two
            if hasattr(message, 'content') and message.content
        ])

    prompt = f"""Analiza si esta pregunta necesita buscar en la base de conocimiento empresarial.
Contexto inmediato (últimos 2 mensajes):
{recent_context if recent_context else 'Sin contexto previo'}
Pregunta actual: {state['question']}
Responde ÚNICAMENTE con este JSON (sin markdown):
{{"needs_retrieval": true, "reason": "razón breve"}}
REGLAS ESTRICTAS:
needs_retrieval = true SIEMPRE que:
  - La pregunta pida información específica (políticas, procedimientos, datos)
  - Mencione documentos, manuales, contratos o información interna
  - Sea una pregunta factual sobre la empresa o sus procesos
  - Haya cualquier duda
needs_retrieval = false SOLO cuando sea OBVIO:
  - Saludos puros: "hola", "gracias", "adiós"
  - Aclaración de lo que la IA dijo en el mensaje inmediato anterior
En caso de duda: needs_retrieval = true"""

    result = llm.invoke([HumanMessage(content=prompt)])

    try:
        content = result.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        decision = json.loads(content.strip())
        needs_retrieval = decision.get("needs_retrieval", True)
        reason = decision.get("reason", "")

    except json.JSONDecodeError:
        needs_retrieval = True
        reason = "Error de parseo, buscando por defecto"

    print(f"[analyze] needs_retrieval={needs_retrieval} | {reason}")

    return {"needs_retrieval": needs_retrieval}


def node_retrieve(state: RAGAgentState, vectorstore: Chroma) -> dict:
    """Busca los chunks más relevantes en ChromaDB"""
    print(f" [retrieve] Buscando: '{state['question'][:60]}...'")

    docs = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.TOP_K_RESULTS}
    ).invoke(state["question"])

    retrieved_texts = [doc.page_content for doc in docs]

    sources = [
        {
            "file": doc.metadata.get("file_name", "desconocida"),
            "page": doc.metadata.get("page", "N/A"),
        }
        for doc in docs
    ]

    print(f" [retrieve] {len(docs)} chunks encontrados")

    for src in sources:
        print(f" -> {src['file']} (pág. {src['page']})")

    return {
        "retrieved_docs": retrieved_texts,
        "sources": sources
    }
