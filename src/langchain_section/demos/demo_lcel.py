"""Demo LCEL"""
from src.langchain_section.core.llm import get_llm
from src.langchain_section.config.settings import settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


if __name__ == "__main__":
    print("="*60)
    print("LangChain LCEL - Fundamentos")
    demo_simple_chain()
    print("="*60)
