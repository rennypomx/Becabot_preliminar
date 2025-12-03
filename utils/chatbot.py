import streamlit as st
import os
from collections import defaultdict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, ServiceUnavailable

# Importar módulo de voz
from utils.voice_input import record_and_transcribe

# ---------------------------------------------------------
#  Crear la cadena de recuperación + generación (RAG)
# ---------------------------------------------------------
def get_context_retriever_chain(vectordb):
    """
    Crea la cadena de recuperación + generación con el modelo Gemini.
    """
    load_dotenv()

    try:
        # Modelo generativo de Gemini (ajustable según tu API)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            max_output_tokens=2048,  # Aumentar límite de tokens de salida
            convert_system_message_to_human=True
        )

        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}  # Aumentado a 15 para mejor cobertura
        )

        prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Eres BecaBot UTPL, un asistente virtual especializado en becas de la Universidad Técnica Particular de Loja. "
     "Eres amable, profesional y siempre útil. "
     "\n\n"
     "Tu base de conocimientos incluye información completa sobre:\n"
     "- Todas las becas disponibles en la UTPL\n"
     "- Requisitos, porcentajes y beneficios de cada beca\n"
     "- Procesos de postulación y renovación\n"
     "- Manuales y procedimientos institucionales\n"
     "\n\n"
     "REGLAS DE CONVERSACIÓN:\n"
     "- MANTÉN CONTINUIDAD: Si ya saludaste al usuario, NO vuelvas a hacerlo.\n"
     "- SALUDO INICIAL: Si es el primer mensaje del usuario, responde: '¡Hola! Soy BecaBot UTPL, tu asistente de becas. ¿En qué puedo ayudarte?'\n"
     "- Revisa el historial para mantener el contexto de la conversación.\n"
     "- Sé natural y conversacional, como si fueras un asesor universitario real.\n"
     "\n\n"
     "REGLAS DE INFORMACIÓN:\n"
     "- USA SOLO la información del sistema que tienes disponible.\n"
     "- NO menciones 'documentos', 'archivos', 'PDFs' ni 'contextos proporcionados'.\n"
     "- Responde como si toda la información estuviera en tu memoria interna.\n"
     "- Cuando cites información, di: 'De acuerdo al sistema de becas UTPL...' o 'Según la información institucional...'\n"
     "- Si NO encuentras información: 'No cuento con esa información en el sistema.'\n"
     "- NUNCA inventes datos. Si no sabes algo, admítelo claramente.\n"
     "\n\n"
     "ESTILO DE RESPUESTA:\n"
     "- Sé claro, directo y profesional.\n"
     "- Estructura bien la información (usa listas cuando sea apropiado).\n"
     "- Enfócate en ser útil y resolver la necesidad del usuario.\n"
     "- Si la pregunta es casual (gracias, adiós, etc.), responde naturalmente.\n\n"
     "Información del sistema:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

        chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        retrieval_chain = create_retrieval_chain(retriever, chain)
        return retrieval_chain

    except (PermissionDenied, ResourceExhausted, ServiceUnavailable):
        st.error(" No se pudo conectar con el modelo Gemini. Verifica tu cuota o API key.")
        return None


# ---------------------------------------------------------
#  Obtener respuesta del modelo
# ---------------------------------------------------------
def get_response(question, chat_history, vectordb, chain_cache=None):
    """
    Genera una respuesta usando el contexto del vector DB.
    """
    chain = chain_cache or get_context_retriever_chain(vectordb)
    if not chain:
        return "No se pudo crear la cadena de recuperación.", []

    try:
        response = chain.invoke({"input": question, "chat_history": chat_history})
        return response["answer"], response["context"]
    except Exception as e:
        st.error(f"Error al generar la respuesta: {str(e)}")
        return "Ocurrió un error al procesar tu consulta.", []


# ---------------------------------------------------------
#  Interfaz de chat con texto y voz
# ---------------------------------------------------------
def chat(chat_history, vectordb):
    """
    Maneja la interacción con el chatbot: texto + voz.
    """
    # Cachear el chain para no recrearlo con cada mensaje
    if "retrieval_chain" not in st.session_state:
        st.session_state.retrieval_chain = get_context_retriever_chain(vectordb)

    # Mostrar historial de chat PRIMERO (para que el usuario vea la conversación continua)
    for message in chat_history:
        role = "AI" if isinstance(message, AIMessage) else "Human"
        with st.chat_message(role):
            st.write(message.content)

    # Interfaz: campo de texto + botón de voz
    col1, col2 = st.columns([4, 1])
    with col1:
        user_query = st.chat_input("Haz una pregunta o usa el micrófono 🎤:")
    with col2:
        # Usar session_state para almacenar la consulta de voz
        if st.button("🎙️ Hablar", help="Presiona para grabar tu voz", use_container_width=True):
            voice_query = record_and_transcribe()
            if voice_query:
                # Guardar en session_state para procesarlo
                st.session_state.voice_query = voice_query
                st.rerun()
    
    # Procesar consulta de voz si existe
    if "voice_query" in st.session_state and st.session_state.voice_query:
        user_query = st.session_state.voice_query
        st.session_state.voice_query = None  # Limpiar después de usar

    if user_query:
        # Mostrar mensaje del usuario inmediatamente
        with st.chat_message("Human"):
            st.write(user_query)
        
        # Generar respuesta con historial y base vectorial
        response, context = get_response(
            user_query, chat_history, vectordb, st.session_state.retrieval_chain
        )

        # Mostrar respuesta del bot
        with st.chat_message("AI"):
            st.write(response)

        # Actualizar historial de conversación
        chat_history.append(HumanMessage(content=user_query))
        chat_history.append(AIMessage(content=response))

        # Mostrar las fuentes del contexto en la barra lateral
        with st.sidebar:
            st.subheader("Documentos Recuperados")
            st.caption("Documentos que el sistema revisó para responder tu pregunta:")
            
            # Separar fuentes por tipo
            pdf_sources = {}
            web_sources = {}
            
            for doc in context:
                metadata = doc.metadata
                source = metadata.get('source', 'Desconocido')
                
                if source.endswith('.pdf'):
                    # Extraer solo el nombre del archivo, no el path completo
                    filename = os.path.basename(source)
                    if filename not in pdf_sources:
                        pdf_sources[filename] = []
                    if 'page' in metadata:
                        pdf_sources[filename].append(metadata['page'])
                else:
                    if source not in web_sources:
                        web_sources[source] = []
                    if 'titulo' in metadata:
                        web_sources[source].append(metadata['titulo'])
            
            # Mostrar PDFs si hay
            if pdf_sources:
                st.write("**Documentos PDF:**")
                for source, pages in pdf_sources.items():
                    unique_pages = sorted(set(map(str, pages)), key=lambda x: int(x) if x.isdigit() else 0)
                    st.write(f"• {source} (páginas: {', '.join(unique_pages)})")
            
            # Mostrar fuentes web si hay
            if web_sources:
                st.write("**Base de Becas Web:**")
                for source, titulos in web_sources.items():
                    unique_titulos = list(set(titulos))
                    if len(unique_titulos) > 0:
                        # Mostrar los nombres de las becas consultadas
                        for titulo in unique_titulos:
                            st.write(f"• {titulo}")
            
            if not pdf_sources and not web_sources:
                st.info("No se recuperaron documentos específicos para esta consulta.")

    return chat_history
