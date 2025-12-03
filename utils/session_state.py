import os
from utils.prepare_vectordb import get_vectorstore

def initialize_session_state_variables(st):
    """
    Inicializa las variables globales de sesión para la aplicación Streamlit.

    Parámetros:
    - st: objeto Streamlit (para acceder a st.session_state)

    Variables inicializadas:
    ├── chat_history: historial de conversación (lista de mensajes)
    ├── uploaded_pdfs: PDFs subidos por el usuario (lista de archivos)
    ├── processed_documents: PDFs ya procesados en la base vectorial
    ├── vectordb: instancia persistente de la base vectorial (Chroma)
    └── previous_upload_docs_length: cantidad de documentos previos
    """

    # --- 1 Asegurar carpeta 'docs' ---
    if not os.path.exists("docs"):
        os.makedirs("docs")

    # --- 2 Leer archivos existentes ---
    upload_docs = os.listdir("docs")

    # --- 3 Variables necesarias ---
    variables = [
        "chat_history",
        "uploaded_pdfs",
        "processed_documents",
        "vectordb",
        "previous_upload_docs_length",
        "voice_query",
    ]

    # --- 4 Inicializar si no existen ---
    for var in variables:
        if var not in st.session_state:
            if var == "chat_history":
                st.session_state.chat_history = []
            elif var == "uploaded_pdfs":
                st.session_state.uploaded_pdfs = []
            elif var == "processed_documents":
                st.session_state.processed_documents = upload_docs
            elif var == "previous_upload_docs_length":
                st.session_state.previous_upload_docs_length = len(upload_docs)
            elif var == "voice_query":
                st.session_state.voice_query = None
            elif var == "vectordb":
                try:
                    # Intenta cargar una base vectorial existente
                    st.session_state.vectordb = get_vectorstore(upload_docs, from_session_state=True)
                except Exception as e:
                    st.warning(f"⚠️ No se pudo cargar la base vectorial existente: {e}")
                    st.session_state.vectordb = None

