import json
import os
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# ============================================================
# 1. Configuración del Navegador
# ============================================================
def configurar_driver():
    options = Options()
    options.add_argument('--headless')  # Ejecutar sin abrir ventana visual
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1920,1080') # Importante para asegurar que se carguen los elementos
    
    # Instalación automática del driver compatible con tu versión de Chrome
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

# ============================================================
# 2. Lógica de Procesamiento de Metadatos (Clases CSS)
# ============================================================
def procesar_metadatos(lista_clases):
    """Traduce las clases CSS a texto legible."""
    clases_set = set(lista_clases)
    
    mapa_tipos = {
        'Excelencia': 'Beca de Excelencia',
        'Inclusión': 'Beca de Inclusión',
        'Estratégica': 'Beca Estratégica',
        'Apoyo': 'Beca de Apoyo Económico',
        'Meritos': 'Méritos Universitarios',
        'Convenios': 'Convenios Institucionales'
    }
    
    mapa_modalidades = {
        'Presencial': 'Presencial',
        'Distancia': 'Abierta y a Distancia',
        'Linea': 'En Línea'
    }

    tipos = [v for k, v in mapa_tipos.items() if k in clases_set]
    modalidades = [v for k, v in mapa_modalidades.items() if k in clases_set]
    
    return tipos, modalidades

# ============================================================
# 3. Parseo Estructurado (La lógica V3 "Clave-Valor")
# ============================================================
def parsear_detalle_estructurado(soup):
    """
    Extrae la información en formato diccionario buscando pares Label-Valor
    típicos de Drupal/UTPL.
    """
    detalles = {}
    
    # Buscamos el contenedor principal
    region = soup.find('div', class_='region-content') or soup.find('div', class_='content')
    if not region:
        return {"Nota": "No se detectó el contenedor principal de contenido."}

    # Estrategia A: Buscar estructura de campos 'field'
    campos = region.find_all('div', class_=lambda x: x and 'field' in x.split())
    
    found_structure = False
    for campo in campos:
        etiqueta_div = campo.find('div', class_='field-label')
        items_div = campo.find('div', class_='field-items')
        
        if etiqueta_div and items_div:
            key = etiqueta_div.get_text(strip=True).rstrip(':')
            # Mantenemos saltos de línea para listas dentro de los valores
            value = items_div.get_text(separator='\n', strip=True)
            detalles[key] = value
            found_structure = True

    # Estrategia B: Tablas HTML
    if not found_structure:
        filas = region.find_all('tr')
        for fila in filas:
            cols = fila.find_all(['td', 'th'])
            if len(cols) >= 2:
                key = cols[0].get_text(strip=True).rstrip(':')
                val = cols[1].get_text(separator='\n', strip=True)
                detalles[key] = val
                found_structure = True

    # Estrategia C: Fallback a texto plano si falla la estructura
    if not found_structure:
        return {"Información General": region.get_text(separator='\n', strip=True)}
        
    return detalles

# ============================================================
# 4. Función Principal de Scraping (Orquestador)
# ============================================================
def scrape_utpl_becas(save_path="knowledge_base/corpus_utpl.json"):
    """
    Función principal para llamar desde tu app.py.
    Realiza el scraping completo y guarda el JSON.
    """
    url_base = "https://becas.utpl.edu.ec/"
    print(f"Iniciando scraping avanzado en {url_base}...")
    
    driver = None
    lista_becas = []

    try:
        driver = configurar_driver()
        driver.get(url_base)
        time.sleep(5) # Espera a que cargue el JS inicial
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # --- PASO 1: OBTENER LISTA DE ENLACES ---
        secciones = {'grado': 'Grado', 'posgrado': 'Posgrado', 'tecnologia': 'Tecnologías'}
        
        for clase_sec, nombre_nivel in secciones.items():
            contenedor = soup.find('div', class_=clase_sec)
            if not contenedor: continue
            
            items = contenedor.find_all('div', class_='item')
            print(f"   -> Procesando sección {nombre_nivel}: {len(items)} becas encontradas.")
            
            for item in items:
                enlace = item.find('a')
                if enlace:
                    url_relativa = enlace.get('href')
                    url_completa = url_base + url_relativa if url_relativa and not url_relativa.startswith('http') else url_relativa
                    
                    # Extraer metadatos de las clases CSS
                    tipos, mods = procesar_metadatos(item.get('class', []))
                    
                    lista_becas.append({
                        "titulo": enlace.get_text(strip=True),
                        "url": url_completa,
                        "nivel": nombre_nivel,
                        "tipos": tipos,
                        "modalidades": mods,
                        "contenido": {} # Placeholder
                    })
        
        # --- PASO 2: ENRIQUECER CON DETALLE (LINK POR LINK) ---
        total = len(lista_becas)
        print(f"📡 Descargando detalles de {total} becas...")
        
        for i, beca in enumerate(lista_becas):
            print(f"   [{i+1}/{total}] {beca['titulo']}")
            try:
                driver.get(beca['url'])
                time.sleep(1.5) # Pausa ética y técnica
                soup_detalle = BeautifulSoup(driver.page_source, 'html.parser')
                
                # Usamos la función de parseo estructurado
                beca['contenido'] = parsear_detalle_estructurado(soup_detalle)
                
            except Exception as e:
                print(f"   ⚠️ Error en {beca['url']}: {e}")
                beca['contenido'] = {"Error": "No se pudo extraer contenido."}

        # --- GUARDADO ---
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(lista_becas, f, ensure_ascii=False, indent=4)

        print(f"✅ Scraping finalizado. Corpus guardado en: {save_path}")
        return lista_becas

    except Exception as e:
        print(f"❌ Error crítico en el scraping: {e}")
        return []
        
    finally:
        if driver:
            driver.quit()

# Bloque para probarlo ejecutando este archivo directamente
if __name__ == "__main__":
    scrape_utpl_becas()