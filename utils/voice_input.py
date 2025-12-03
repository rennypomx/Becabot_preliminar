import streamlit as st
import speech_recognition as sr
import time

def record_and_transcribe(timeout=5, phrase_time_limit=10):
    """
    Graba audio del micrófono y devuelve el texto transcrito usando el motor de Google.
    
    Parámetros:
    - timeout: Tiempo máximo de espera antes de comenzar a grabar (segundos)
    - phrase_time_limit: Tiempo máximo de grabación (segundos)
    
    Requiere conexión a Internet y micrófono activo.
    Usa Google Speech Recognition API (gratuita, requiere internet).
    
    Alternativa futura: OpenAI Whisper API para mayor precisión en español.
    """
    recognizer = sr.Recognizer()
    
    # Configuración optimizada para mejor precisión
    recognizer.energy_threshold = 300  # Sensibilidad del micrófono (valor por defecto)
    recognizer.dynamic_energy_threshold = True  # Se ajusta automáticamente
    recognizer.pause_threshold = 0.8  # Pausa antes de finalizar
    
    try:
        mic = sr.Microphone()
    except OSError as e:
        st.error("❌ No se detectó ningún micrófono. Verifica que esté conectado.")
        return None

    with mic as source:
        # Ajuste mejorado de ruido ambiente
        with st.spinner("🔊 Ajustando al ruido ambiente..."):
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
        
        # Indicador de grabación
        st.warning(f"🎙️ **GRABANDO** - Habla ahora (máx. {phrase_time_limit}s)")
        
        try:
            # Grabar con timeout
            audio = recognizer.listen(
                source, 
                timeout=timeout,
                phrase_time_limit=phrase_time_limit
            )
            
        except sr.WaitTimeoutError:
            st.error("Tiempo de espera agotado. No se detectó audio.")
            return None

    # Transcripción
    try:
        with st.spinner("Transcribiendo tu voz..."):
            # Usar Google Speech Recognition con español de España
            text = recognizer.recognize_google(audio, language="es-ES")
            
        st.success(f"**Transcripción:** {text}")
        return text

    except sr.UnknownValueError:
        st.error("❌ No se entendió lo que dijiste. Intenta hablar más claro y cerca del micrófono.")
        return None

    except sr.RequestError as e:
        st.error(f"❌ Error de conexión con Google Speech API: {e}")
        st.info("💡 Verifica tu conexión a internet.")
        return None
        
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
        return None
