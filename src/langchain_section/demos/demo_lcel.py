"""Demo LCEL"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.langchain_section.core.llm import get_llm
from src.langchain_section.config.settings import settings


llm = get_llm()


# código aquí
def demo_simple_chain() -> None:
    """Cadena simple"""

    # ChatPromptTemplate (role, contenido)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Eres un experto en {tema}. Responde de forma concisa, máximo 3 oraciones"),
        ("human", "{pregunta}")
    ])

    # StrOutParser
    parser = StrOutputParser()

    # Cadena (Chain)
    chain = prompt | llm | parser

    response = chain.invoke({
        "tema": "Python",
        "pregunta": "¿Qué es un decorador?"
    })

    print("Simple Chain")
    print(response)
    print()


def demo_steps_inspection() -> None:
    """Invoca cada componente"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente técnico."),
        ("human", "{pregunta}")
    ])

    parser = StrOutputParser()

    input_message = {"pregunta": "¿Qué es una API REST?"}

    # Paso 1 prompt
    messages = prompt.invoke(input_message)
    print("Paso 1 - Prompt output")
    print(f" Tipo: {type(messages).__name__}")
    print(f" Mensajes: {messages.messages}")

    # Paso 2 LLM -> AIMessage
    ai_message = llm.invoke(messages)
    print("Paso 2 - LLM output")
    print(f" Tipo: {type(ai_message).__name__}")
    print(f" AIMessage: {ai_message} ")
    print(f" Contenido: {ai_message.content[:100]}...")

    # Paso 3: Parser extraer texto
    text = parser.invoke(ai_message)
    print("Paso 3 - Parser output")
    print(f" Tipo: {type(text).__name__}")
    print(f" Texto: {text[:100]}...")
    print()


if __name__ == "__main__":
    print("="*60)
    print("LangChain LCEL - Fundamentos")
    # demo_simple_chain()
    demo_steps_inspection()
    print("="*60)
