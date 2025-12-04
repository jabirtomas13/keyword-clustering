import streamlit as st
import pandas as pd
import numpy as np
import time
import spacy

# Importaciones para IA (Descomentar cuando tengas las API keys activas)
# import openai
# import google.genai

# --- CONFIGURACIÓN DE PÁGINA ---
# Debe ser siempre la primera línea de comando de Streamlit
st.set_page_config(
    page_title="AI Text Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stTextArea textarea { font-size: 16px !important; border-radius: 10px; border: 1px solid #e0e0e0; }
    .stButton>button { border-radius: 20px; font-weight: 600; width: 100%; }
    h1 { color: #1E1E1E; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# --- GESTIÓN DE MODELOS (CACHÉ Y CARGA) ---

@st.cache_resource
def load_spacy_model():
    """
    Carga el modelo de SpaCy solo una vez para optimizar rendimiento.
    Intenta cargar el modelo en español, si falla, usa uno genérico o vacío.
    """
    try:
        # Intentamos cargar el modelo instalado via requirements.txt
        nlp = spacy.load("es_core_news_sm")
        return nlp
    except OSError:
        # Fallback si no se encuentra el modelo (útil para pruebas locales sin instalación completa)
        st.warning("Modelo 'es_core_news_sm' no encontrado. Asegúrate de instalarlo.")
        return None

# --- FUNCIONES DE LÓGICA (CORE) ---

def procesar_texto_simulado(texto, operacion):
    """
    Función PLACEHOLDER que simula una llamada a OpenAI/Gemini.
    Sustituir esto con tu lógica real de 'client.chat.completions.create' etc.
    """
    time.sleep(1.5) # Simular latencia de red
    
    if operacion == "resumir":
        return f"**Resumen AI:** El texto proporcionado trata sobre '{texto[:30]}...'. Se identifican puntos clave sobre tecnología y negocios."
    elif operacion == "traducir":
        return f"**Traducción:** {texto} (Simulación de traducción al inglés)."
    elif operacion == "analizar":
        return "Análisis general completado. Tono detectado: Formal."
    return "Operación no reconocida."

def analizar_entidades_spacy(texto, nlp_model):
    """Procesa el texto con SpaCy para extraer entidades"""
    if nlp_model is None:
        return pd.DataFrame({"Info": ["Modelo SpaCy no cargado"]})
    
    doc = nlp_model(texto)
    data = []
    for ent in doc.ents:
        data.append({
            "Entidad": ent.text,
            "Etiqueta": ent.label_,
            "Descripción": spacy.explain(ent.label_)
        })
    
    if not data:
        return pd.DataFrame({"Resultado": ["No se encontraron entidades"]})
        
    return pd.DataFrame(data)

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.title("⚙️ Configuración")
    
    st.markdown("### 🧠 Motor de IA")
    model_provider = st.selectbox(
        "Seleccionar Modelo",
        ["OpenAI (GPT-4)", "Google Gemini", "Simulación (Demo)"],
        index=2, # Por defecto en Demo para que funcione al desplegar sin keys
        help="Elige 'Simulación' para probar la UI sin gastar créditos."
    )
    
    api_key = st.text_input("API Key", type="password", placeholder="sk-...", help="Necesaria para OpenAI/Gemini")
    
    st.markdown("---")
    st.markdown("### 🛠 Herramientas")
    enable_ner = st.toggle("Extraer Entidades (NER)", value=True)
    enable_sentiment = st.toggle("Analizar Sentimiento", value=False)
    
    creativity = st.slider("Creatividad", 0.0, 1.0, 0.5)

# --- INTERFAZ PRINCIPAL ---

col1, col2 = st.columns([3, 1])
with col1:
    st.title("AI Text Studio")
    st.markdown("Plataforma de análisis inteligente de texto.")
with col2:
    st.metric(label="Estado", value="Listo", delta="Online")

st.markdown("---")

# Input
input_col, btn_col = st.columns([3, 1])
with input_col:
    user_text = st.text_area("Texto de entrada", height=200, placeholder="Escribe aquí para analizar...")

with btn_col:
    st.markdown("#### Acciones")
    btn_analyze = st.button("🔍 Analizar Todo", type="primary")
    btn_summary = st.button("📝 Resumir")
    btn_translate = st.button("🌐 Traducir EN")
    
    if user_text:
        st.caption(f"Palabras: {len(user_text.split())}")

# --- PROCESAMIENTO Y RESULTADOS ---

if btn_analyze or btn_summary or btn_translate:
    if not user_text:
        st.warning("⚠️ Por favor ingresa un texto primero.")
    else:
        # Cargar recursos
        nlp = load_spacy_model()
        
        with st.spinner("Procesando solicitud..."):
            # Lógica de decisión
            if btn_summary:
                resultado_texto = procesar_texto_simulado(user_text, "resumir")
                active_tab = 0
            elif btn_translate:
                resultado_texto = procesar_texto_simulado(user_text, "traducir")
                active_tab = 0
            else:
                resultado_texto = procesar_texto_simulado(user_text, "analizar")
                active_tab = 1

            # Renderizado de Resultados en Pestañas
            st.markdown("### 📊 Resultados")
            tab1, tab2, tab3 = st.tabs(["Respuesta IA", "Entidades (NER)", "Datos Técnicos"])
            
            with tab1:
                st.success("Proceso finalizado")
                st.markdown(resultado_texto)
            
            with tab2:
                if enable_ner and nlp:
                    st.markdown("#### Entidades Detectadas")
                    df_ents = analizar_entidades_spacy(user_text, nlp)
                    st.dataframe(df_ents, use_container_width=True)
                elif not enable_ner:
                    st.info("Reconocimiento de entidades desactivado.")
                else:
                    st.error("No se pudo cargar el modelo de SpaCy.")
            
            with tab3:
                st.json({
                    "modelo": model_provider,
                    "caracteres": len(user_text),
                    "timestamp": time.strftime("%H:%M:%S")
                })

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>AI Text Studio v1.0</div>", unsafe_allow_html=True)
