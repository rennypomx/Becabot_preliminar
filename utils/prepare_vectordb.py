import os
import json
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import chromadb

# ============================================================
# 🔧 Configuración del entorno
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Fuerza CPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*telemetry.*")

# ============================================================
# Función para extraer texto de PDFs
# ============================================================
def extract_pdf_text(pdfs):
    docs = []
    # Aseguramos que exista la carpeta docs, si no, retornamos lista vacía
    if not os.path.exists("docs"):
        print("⚠️ La carpeta 'docs' no existe.")
        return docs

    for pdf in pdfs:
        pdf_path = os.path.join("docs", pdf)
        try:
            docs.extend(PyPDFLoader(pdf_path).load())
            print(f"Texto extraído de {pdf}")
        except Exception as e:
            print(f"⚠️ Error al procesar {pdf}: {e}")
    return docs


# ============================================================
# Función para extraer texto de JSON del scraping (ACTUALIZADA)
# ============================================================
def extract_json_text(json_path="knowledge_base/corpus_utpl.json"):
    docs = []
    if not os.path.exists(json_path):
        print(f"⚠️ No se encontró el archivo JSON en {json_path}")
        return docs

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            print(f"📂 Procesando {len(data)} becas del archivo JSON...")

            for item in data:
                # 1. Extraer campos principales
                titulo = item.get("titulo", "Beca sin título")
                url = item.get("url", "")
                nivel = item.get("nivel", "General")
                
                # Convertimos listas a strings para el texto
                tipos = ", ".join(item.get("tipos", []))
                modalidades = ", ".join(item.get("modalidades", []))
                
                # 2. Aplanar el diccionario de contenido
                # Convertimos {"Requisitos": "X", "Porcentaje": "Y"} a texto plano
                contenido_raw = item.get("contenido", {})
                contenido_texto = ""
                
                if isinstance(contenido_raw, dict):
                    for clave, valor in contenido_raw.items():
                        # Limpiamos saltos de línea excesivos
                        valor_limpio = str(valor).replace('\n', ' ').strip()
                        contenido_texto += f"- {clave}: {valor_limpio}\n"
                else:
                    # Fallback si por alguna razón llega como string
                    contenido_texto = str(contenido_raw)

                # 3. Construir el Page Content (Lo que leerá la IA)
                # Estructuramos el texto para darle contexto semántico
                page_content = f"""
                TÍTULO DE LA BECA: {titulo}
                NIVEL ACADÉMICO: {nivel}
                TIPO: {tipos}
                MODALIDAD: {modalidades}
                ENLACE: {url}

                DETALLES, REQUISITOS Y BENEFICIOS:
                {contenido_texto}
                """

                # 4. Crear el Documento con Metadatos
                doc = Document(
                    page_content=page_content,
                    metadata={
                        "source": "corpus_utpl.json",
                        "titulo": titulo,
                        "url": url,
                        "nivel": nivel,
                        "tipo": tipos  # Chroma prefiere strings en metadatos
                    }
                )
                docs.append(doc)
                
        print(f"Se cargaron exitosamente {len(docs)} documentos desde el JSON.")
        
    except Exception as e:
        print(f"❌ Error crítico al leer el JSON: {e}")
        
    return docs


# ============================================================
# División del texto en fragmentos
# ============================================================
def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,     # Aumentado para evitar fragmentar becas individuales
        chunk_overlap=300,   # Mayor overlap para preservar contexto
        separators=["\n\n", "\nTÍTULO DE LA BECA:", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)


# ============================================================
# Generación o carga de la base vectorial
# ============================================================
def get_vectorstore(pdfs, from_session_state=False):
    load_dotenv()

    # Usamos un modelo de embeddings mejorado
    # Configuración especial para evitar el error de meta tensors
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Modelo mejor que L3
        model_kwargs={
            "device": "cpu",
            "trust_remote_code": True
        },
        encode_kwargs={"normalize_embeddings": True}
    )

    persist_dir = "Vector_DB - Documents"

    if from_session_state and os.path.exists(persist_dir):
        try:
            settings = chromadb.config.Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                chroma_telemetry_impl="none"
            )
            client = chromadb.PersistentClient(path=persist_dir, settings=settings)
            vectordb = Chroma(
                client=client,
                embedding_function=embedding
            )
            print("Base vectorial cargada desde el disco.")
            return vectordb
        except Exception as e:
            print(f"⚠️ Error al cargar existente, se regenerará: {e}")

    # Regeneración completa
    print("Iniciando regeneración de base vectorial...")
    
    # 1. Cargar PDFs
    docs_pdf = extract_pdf_text(pdfs)
    
    # 2. Cargar JSON (Nueva lógica)
    docs_json = extract_json_text("knowledge_base/corpus_utpl.json")

    all_docs = docs_pdf + docs_json
    
    if not all_docs:
        print("⚠️ No hay documentos para procesar.")
        return None

    # 3. Dividir en chunks
    chunks = get_text_chunks(all_docs)
    print(f"Total de fragmentos generados: {len(chunks)}")

    # 4. Crear Vector Store
    try:
        settings = chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            chroma_telemetry_impl="none"
        )
        client = chromadb.PersistentClient(path=persist_dir, settings=settings)
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            client=client
        )
        print("Base vectorial creada y guardada correctamente en disco.")
        return vectordb
    except Exception as e:
        print(f"❌ Error al crear la base vectorial en Chroma: {e}")
        return None


# ============================================================
# Ejecución directa
# ============================================================
if __name__ == "__main__":
    print("Script de preparación de vectores iniciado...")
    
    # Listar PDFs si existen
    pdfs = []
    if os.path.exists("docs"):
        pdfs = [f for f in os.listdir("docs") if f.endswith(".pdf")]
    
    vectordb = get_vectorstore(pdfs, from_session_state=False)
    
    if vectordb:
        print("Proceso completado exitosamente.")
    else:
        print("❌ El proceso finalizó con errores.")