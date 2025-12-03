import streamlit as st
import os
from utils.save_docs import save_docs_to_vectordb
from utils.session_state import initialize_session_state_variables
from utils.prepare_vectordb import get_vectorstore
from utils.chatbot import chat
# CAMBIO 1: Importamos la nueva función de scraping de becas
from utils.web_scraper import scrape_utpl_becas 

class ChatApp:
    """
    Aplicación Streamlit para chatear con documentos PDF y un corpus web (RAG con LangChain + Gemini).
    """

    def __init__(self):
        """
        Inicializa la aplicación:
        - Configura la página
        - Crea carpeta 'docs' si no existe
        - Verifica si existe el corpus, si no, lo descarga.
        - Inicializa las variables de sesión
        """
        # Configuración inicial de la página
        st.set_page_config(page_title="Chat Becas UTPL", layout="wide")
        st.title("Bienvenido a BecaBot UTPL")

        # Asegurar carpetas necesarias
        if not os.path.exists("docs"):
            os.makedirs("docs")
        
        corpus_path = "knowledge_base/corpus_utpl.json"

        # CAMBIO 2: Lógica inteligente de Scraping
        # Solo scrapeamos automáticamente si el archivo NO existe.
        # Evitamos ejecutar Selenium en cada recarga de página.
        if not os.path.exists(corpus_path):
            with st.spinner("Inicializando: Descargando información de Becas UTPL..."):
                scrape_utpl_becas(save_path=corpus_path)
                st.success("Corpus base descargado.")
        
        # Inicializar variables de sesión
        initialize_session_state_variables(st)
        self.docs_files = st.session_state.processed_documents

    def run(self):
        """
        Ejecuta la aplicación principal.
        """
        upload_docs = os.listdir("docs")

        # 📂 Sidebar — gestión de documentos y base de conocimiento
        with st.sidebar:
            st.header("Gestión de Conocimiento")
            
            # --- INFORMACIÓN DEL CHAT ---
            num_messages = len(st.session_state.chat_history)
            if num_messages > 0:
                st.info(f"Conversación activa: {num_messages // 2} intercambios")
            
            # --- BOTÓN PARA LIMPIAR HISTORIAL ---
            if st.button("Nueva Conversación", help="Limpia el historial del chat"):
                st.session_state.chat_history = []
                if "retrieval_chain" in st.session_state:
                    del st.session_state.retrieval_chain
                st.rerun()
            
            st.divider()
            
            
            # --- SECCIÓN 1: ACTUALIZACIÓN WEB ---
            st.subheader("Información Web (Becas)")
            # Botón manual para forzar la actualización del scraping
            if st.button("Actualizar Becas (Web Scraping)"):
                with st.spinner("⏳ Conectando con becas.utpl.edu.ec... esto puede tardar un poco..."):
                    scrape_utpl_becas() # Ejecuta el scraping
                    # Forzamos la regeneración de la base vectorial
                    st.session_state.vectordb = get_vectorstore(upload_docs, from_session_state=False)
                    st.success("¡Información de becas actualizada!")

            st.divider()

            # --- SECCIÓN 2: DOCUMENTOS PDF ---
            st.subheader("Tus Documentos PDF")
            if upload_docs:
                st.write("Archivos cargados:")
                st.caption(", ".join(upload_docs))
            else:
                st.info("No hay PDFs cargados.")

            # Subir nuevos PDFs
            pdf_docs = st.file_uploader(
                "Sube archivos PDF extra",
                type=['pdf'],
                accept_multiple_files=True
            )

            # Procesar nuevos PDFs
            if pdf_docs:
                if st.button("Procesar PDFs"):
                    # Guardar archivos en carpeta docs
                    files_saved = save_docs_to_vectordb(pdf_docs, upload_docs)
                    
                    if files_saved:
                        # Actualizar lista de archivos
                        upload_docs = os.listdir("docs")
                        
                        # Regenerar base vectorial con TODOS los PDFs
                        with st.spinner("Actualizando base de conocimiento..."):
                            st.session_state.vectordb = get_vectorstore(upload_docs, from_session_state=False)
                            # Forzar recreación del chain para usar la nueva base vectorial
                            if "retrieval_chain" in st.session_state:
                                del st.session_state.retrieval_chain
                            st.session_state.previous_upload_docs_length = len(upload_docs)
                            st.success("PDFs integrados a la base de conocimiento.")
                            st.rerun()  # Recargar para actualizar la lista de archivos

        # 💬 LÓGICA DEL CHAT
        # Si existen documentos O existe el corpus de becas (que siempre debería existir tras el init)
        # Habilitamos el chat.
        corpus_exists = os.path.exists("knowledge_base/corpus_utpl.json")
        
        if self.docs_files or st.session_state.uploaded_pdfs or corpus_exists:
            
            # Carga inicial de la base vectorial si no está en sesión
            if st.session_state.vectordb is None:
                with st.spinner("Cargando cerebro del chatbot..."):
                    # Si hay PDFs en la carpeta docs, forzar regeneración para incluirlos
                    should_regenerate = len(upload_docs) > 0
                    st.session_state.vectordb = get_vectorstore(upload_docs, from_session_state=not should_regenerate)

            # Si por alguna razón la carga falló, intentar regenerar
            if st.session_state.vectordb is None:
                 st.session_state.vectordb = get_vectorstore(upload_docs, from_session_state=False)

            # Ejecutar el chat
            if st.session_state.vectordb:
                st.session_state.chat_history = chat(st.session_state.chat_history, st.session_state.vectordb)
            else:
                st.error("No se pudo iniciar la base de datos vectorial.")

        else:
            st.info("Esperando datos para iniciar...")

# Punto de entrada principal
if __name__ == "__main__":
    app = ChatApp()
    app.run()