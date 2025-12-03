import streamlit as st
import os
from utils.prepare_vectordb import get_vectorstore

def save_docs_to_vectordb(pdf_docs, upload_docs):
    """
    Guarda los documentos PDF subidos en la carpeta 'docs' y actualiza la base vectorial (ChromaDB).

    Parámetros:
    - pdf_docs (list): Archivos PDF subidos actualmente.
    - upload_docs (list): Nombres de los documentos ya existentes en /docs.
    """
    # --- 1 Verificar carpeta destino ---
    if not os.path.exists("docs"):
        os.makedirs("docs")

    # --- 2 Filtrar solo los nuevos archivos ---
    new_files = [pdf for pdf in pdf_docs if pdf.name not in upload_docs]
    new_files_names = [pdf.name for pdf in new_files]

    # --- 3 Guardar archivos nuevos ---
    if new_files:
        try:
            # Guardar PDFs localmente
            for pdf in new_files:
                pdf_path = os.path.join("docs", pdf.name)
                with open(pdf_path, "wb") as f:
                    f.write(pdf.getvalue())
                if pdf.name not in st.session_state.uploaded_pdfs:
                    st.session_state.uploaded_pdfs.append(pdf.name)
            
            st.success(f" {len(new_files)} archivo(s) guardado(s): " + ", ".join(new_files_names))
            return True

        except Exception as e:
            st.error(f"❌ Error al guardar los archivos: {e}")
            return False
    else:
        st.info("ℹ No hay nuevos documentos. Todos los archivos ya están guardados.")
        return False
