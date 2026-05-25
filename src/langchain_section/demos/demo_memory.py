
from turtle import back

from langchain_core.runnables import RunnableWithMessageHistory
from langsmith import expect

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


def run_chat_session(
    chatbot: RunnableWithMessageHistory,
    backend: BaseMemoryBackend,
    session_id: str
) -> None:
    print(f"\nSesión activa: {session_id}")

    messages = backend.get_history(session_id).messages

    if messages:
        print(
            f"✅ Retomando la conversación ({len(messages)}) mensajes previos")
    else:
        print("Nueva conversación")

    print("Comandos: 'historial' | 'limpiar' | 'sesiones' | 'salir' \n")

    while True:
        try:
            user_input = input("Tú: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "salir":
                print("¡Hasta luego! 👋 Historial guardado")
                break

            if user_input.lower() == "historial":
                messages = backend.get_history(session_id).messages

                if not messages:
                    print(" [Historial vacío]\n")
                    continue
                print(f"\nÚltimos mensajes de '{session_id}'")

                for message in messages[-6:]:
                    rol = "Tú" if message.type == "human" else "IA"
                    print(f" {rol}: {message.content[:70]}...")
                print()
                continue

            if user_input.lower() == "limpiar":
                backend.clear_history(session_id)
                print("HISTORIAL BORRADO.\n")
                continue

            if user_input.lower() == "sesiones":
                sessions = backend.list_sessions()
                print(f"\n Sesiones disponibles: {sessions}\n")
                continue

            response = chatbot.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            print(f"\nIA: {response}\n")
        except KeyboardInterrupt:
            print("¡Hasta luego! 👋")
            break
        except Exception as e:
            print(f"Error: {e}\n")
