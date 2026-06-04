# Python: Inteligencia Artificial Aplicada

Repositorio oficial del curso **Python: Inteligencia Artificial Aplicada**, disponible en **DevTalles**.

En este curso construiremos ejemplos prácticos usando Python para trabajar con inteligencia artificial aplicada, modelos de lenguaje, prompts, bases de datos vectoriales, embeddings, LangChain, memoria conversacional, RAG, agentes, LangGraph, etc.

Este curso se encuentra en dos repositorios: 

Sección 1 - 6 : [En el siguiente repositorio (click aquí)](https://github.com/ricardocuellar/devtalles-python-inteligencia-artificial-aplicada/tree/section-2-fundamentos)

Sección 7 - 10: Este repositorio

> Este repositorio acompaña el avance del curso.  
> Cada rama contiene el código correspondiente al avance de una sección específica.

----------

## Sobre el curso

Este curso forma parte del catálogo de **DevTalles** y está diseñado para aprender a construir aplicaciones reales usando Python e inteligencia artificial aplicada.

A lo largo del curso trabajaremos con herramientas modernas del ecosistema de IA, incluyendo:

-   Modelos de lenguaje.
    
-   Prompts.
    
-   OpenAI API.
    
-   LangChain.
    
-   Memoria conversacional.
    
-   Carga y procesamiento de documentos.
    
-   Embeddings.
    
-   Vector stores.
    
-   RAG.
    
-   Agentes.
    
-   LangGraph.
  
   
El objetivo es que aprendas a construir soluciones prácticas de IA usando Python, desde ejemplos simples hasta flujos más completos y estructurados.

----------

## ¿Qué aprenderás?

Durante el desarrollo del proyecto aprenderás a:

-   Configurar un proyecto de Python para trabajar con IA.
    
-   Consumir modelos de lenguaje desde Python.
    
-   Crear prompts reutilizables.
    
-   Construir cadenas con LangChain.
    
-   Manejar memoria conversacional.
    
-   Cargar documentos reales.
    
-   Dividir documentos en fragmentos.
    
-   Generar embeddings.
    
-   Guardar información en un vector store.
    
-   Crear sistemas RAG para consultar documentos.
    
-   Construir agentes con herramientas.
    
-   Orquestar flujos con LangGraph.
        
-   Organizar un proyecto de IA aplicada con buenas prácticas.
    

----------

## Requisitos previos

Antes de iniciar, asegúrate de tener instalado:

-   Python 3.12 o superior.
    
-   Git.
    
-   Visual Studio Code o tu editor favorito.
    
-   Una terminal o consola.
    
-   Una API Key del proveedor de IA usado durante el curso.
    
-   Conocimientos básicos de Python.

-   Conocimientos de bases de datos (deseable)

- Docker
    

----------

## Clonar el repositorio

```bash
git clone https://github.com/ricardocuellar/devtalles-python-inteligencia-artificial-aplicada
cd devtalles-python-ia-aplicada-langchain

```

----------

## Instalar dependencias

Este repositorio usa `uv`, ejecuta:
```bash
uv sync

```

----------

## Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto.

Puedes tomar como base el archivo de ejemplo:

```bash
cp .env.example .env

```

Luego agrega tus propias credenciales:

```env
OPENAI_API_KEY=tu_api_key_aqui

```

> Importante: nunca subas tus API Keys reales a GitHub.  
> El archivo `.env` debe mantenerse fuera del repositorio mediante `.gitignore`.

----------

## Estructura general del proyecto

La estructura puede cambiar conforme avanza el curso, pero de forma general el proyecto se ve así:

```txt
.
├── data/
│   └── chromadb/              
|   └── documents/          # Documentos usados para pruebas de RAG
├── src/									  # Código principal del curso
│   └── chains/            
│   └── config/
│   └── core/
│   └── demos/
│   └── graphs/
│   └── memory/
├── .env.example            # Ejemplo de variables de entorno
├── .gitignore
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock


```

----------

## Cómo ejecutar los ejemplos

Dependiendo de la sección del curso, encontrarás distintos scripts o módulos ejecutables.

Ejemplo:

```bash
python -m src.langchain_section.demos.demo_lcel

```

Revisa la sección correspondiente del curso para conocer el comando exacto que se debe ejecutar en cada etapa.

----------

# Uso de ramas en este repositorio

Este repositorio está organizado por ramas.

Cada rama representa el avance del proyecto en una sección específica del curso.

Esto permite que puedas:

-   Revisar el código correspondiente a una sección concreta.
    
-   Comparar tu avance con el código del curso.
    
-   Recuperarte si cometiste un error.
    
-   Ver cómo evoluciona el proyecto paso a paso.
    
-   Estudiar una sección sin depender del código final completo.
    

----------

## Ver todas las ramas disponibles

```bash
git branch -a

```

Este comando mostrará las ramas locales y remotas del repositorio.

----------

## Descargar las ramas remotas

Si no ves todas las ramas disponibles, ejecuta:

```bash
git fetch --all

```

----------

## Cambiarte a una rama específica

```bash
git checkout nombre-de-la-rama

```

Ejemplo:

```bash
git checkout seccion-03-langchain-basico

```

----------

## Volver a la rama principal

```bash
git checkout main

```

----------

## Flujo recomendado para estudiantes

Te recomiendo trabajar de esta forma:

1.  Clona el repositorio.
    
2.  Instala las dependencias.
    
3.  Crea tu archivo `.env`.
    
4.  Revisa en qué sección del curso estás.
    
5.  Cambia a la rama correspondiente.
    
6.  Ejecuta los ejemplos de esa sección.
    
7.  Compara tu código con el de la rama si algo no funciona.
    

Ejemplo:

```bash
git fetch --all
git checkout seccion-05-rag

```

----------

## Convención de ramas

Las ramas del repositorio pueden seguir una convención similar a esta:

```txt
seccion-7-langchain-intro
seccion-8-langchain-memory
seccion-9-langchain-rag
seccion-10-langgraph

```

> Los nombres exactos pueden variar.  
> Usa `git branch -a` para ver las ramas reales disponibles en el repositorio.


Acceso directo a las ramas: 
- [Sección 7: Introducción a LangChain](https://github.com/ricardocuellar/devtalles-python-ia-aplicada-langchain/tree/section-7-langchain-intro)
- [Sección 8: LangChain memoria persistente](https://github.com/ricardocuellar/devtalles-python-ia-aplicada-langchain/tree/section-8-langchain-memory)
- [Sección 9: LangChain + RAG)](https://github.com/ricardocuellar/devtalles-python-ia-aplicada-langchain/tree/section-9-langchain-rag)
- [Sección 10: LangGraph](https://github.com/ricardocuellar/devtalles-python-ia-aplicada-langchain/tree/section-10-langgraph)

----------

## ¿Qué hago si mi código no funciona?

Si tienes errores, revisa lo siguiente:

-   Estás en la rama correcta.
    
-   Activaste tu entorno virtual.
    
-   Instalaste las dependencias.
    
-   Creaste tu archivo `.env`.
    
-   Agregaste correctamente tus variables de entorno.
    
-   Estás ejecutando los comandos desde la raíz del proyecto.
    
-   Tu versión de Python es compatible.
    
-   El módulo o archivo que quieres ejecutar existe en la rama actual.
    

También puedes comparar tu código con el código de la rama correspondiente a la sección del curso.

----------

## Comandos útiles de Git

### Ver la rama actual

```bash
git branch

```

### Ver el estado del proyecto

```bash
git status

```

### Descargar últimos cambios

```bash
git pull

```

### Ver historial de commits

```bash
git log --oneline

```

### Cambiar de rama descartando cambios locales

Usa este comando solo si estás seguro de que no necesitas conservar tus cambios locales:

```bash
git reset --hard
git checkout nombre-de-la-rama

```

----------

## Buenas prácticas aplicadas en el proyecto

Durante el curso se busca aplicar buenas prácticas como:

-   Separar configuración sensible usando variables de entorno.
    
-   Mantener las API Keys fuera del repositorio.
    
-   Organizar el código por módulos.
    
-   Crear ejemplos progresivos.
    
-   Separar responsabilidades dentro del proyecto.
    
-   Usar nombres claros para archivos, funciones y carpetas.
    
-   Mantener demos pequeñas y fáciles de ejecutar.
    
-   Usar ramas para representar el avance del curso.
    
-   Construir el proyecto paso a paso en lugar de mostrar solo el resultado final.
    

----------

## Notas importantes

Este repositorio es material de apoyo para el curso **Python: Inteligencia Artificial Aplicada** de **DevTalles**.

No todas las ramas representan una aplicación final completa.  
Algunas ramas contienen código parcial o versiones intermedias del proyecto porque están diseñadas para explicar conceptos de forma progresiva.

Si estás tomando el curso, lo ideal es seguir el orden de las clases y usar las ramas como referencia cuando necesites comparar tu avance.

----------

## Curso

Este repositorio acompaña el curso:

**Python: Inteligencia Artificial Aplicada**  
Disponible en: **DevTalles**
Enlace al curso: [Curso en Devtalles](https://cursos.devtalles.com/courses/python-ia-aplicada?coupon=IADEV26)

----------

## Autor

**Ricardo Cuéllar**

Desarrollador de software e instructor.

GitHub: [@ricardocuellar](https://github.com/ricardocuellar)

----------

## Licencia

Este repositorio forma parte del material educativo del curso.

El código está pensado para fines de aprendizaje y acompañamiento del curso.  
Antes de reutilizar, distribuir o publicar este material como propio, revisa los términos correspondientes de la plataforma y del autor.
