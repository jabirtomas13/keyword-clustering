#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword Clustering App – SEO & EEAT High-End UI Edition
-------------------------------------------------------
Integración completa:
1. UX/UI Moderna (CSS corregido para legibilidad).
2. Lógica SEO Avanzada (SBERT Embeddings, Gemini EEAT Prompt).
3. Ajustes de visualización en tablas.
"""
import os
import re
import unicodedata
import json
import time
from typing import List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --- CONFIGURACIÓN DE PÁGINA (UI) ---
st.set_page_config(
    page_title="SEO Content Clusterizer (EEAT)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    /* Fondo general */
    .main { background-color: #f8f9fa; }
    
    /* Botones primarios */
    .stButton>button { border-radius: 8px; font-weight: 600; text-transform: none; padding: 0.5rem 1rem; }
    .stButton>button[kind="primary"] { background-color: #ff4b4b; border: none; box-shadow: 0 4px 14px 0 rgba(255, 75, 75, 0.39); }
    
    /* Tipografía */
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #0e1117; }
    
    /* Tarjetas de Métricas (Fix de legibilidad) */
    .stMetric { 
        background-color: #ffffff !important; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0; 
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    /* Forzar color de texto oscuro dentro de las métricas para evitar blanco sobre blanco */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #31333F !important;
    }
    
    /* Status widget */
    div[data-testid="stStatusWidget"] { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Imports perezosos (Lazy Imports)
# ------------------------
def _lazy_import_spacy():
    try:
        import spacy
        return spacy
    except Exception:
        return None

def _lazy_import_sbert():
    """Intenta importar sentence_transformers para embeddings de alta calidad."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        return None
    except Exception:
        return None

def _lazy_import_gemini():
    try:
        from google import genai
        if hasattr(genai, 'Client'):
            return genai
        return None
    except ImportError:
        return None
    except Exception:
        return None

# ------------------------
# Utilidades de Normalización
# ------------------------
_SMARTS = {
    "\u2018": "'", "\u2019": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u00A0": " ",
}
_SMARTS_TRANS = str.maketrans(_SMARTS)
_EMOJI_RE = re.compile("[\U0001F300-\U0001F6FF\U0001F700-\U0001FAFF\U00002700-\U000027BF\U0001F900-\U0001F9FF]")
_ALLOWED_RE = re.compile(r"[^a-z0-9\-\' ]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

def strip_diacritics(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))

def normalize_keyword(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    if not s: return ""
    s = s.translate(_SMARTS_TRANS)
    s = _EMOJI_RE.sub(" ", s)
    s = strip_diacritics(s)
    s = s.lower()
    s = _ALLOWED_RE.sub(" ", s)
    s = re.sub(r"\b[-']+|[-']+\b", " ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s

def normalize_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").map(normalize_keyword)

# ------------------------
# Embeddings: SpaCy & SBERT
# ------------------------
@st.cache_resource
def build_spacy_pipeline(lang_choice: str):
    spacy = _lazy_import_spacy()
    if not spacy: return None
    map_md = {
        "auto-multi (xx)": "xx_ent_wiki_sm",
        "english (en)": "en_core_web_md",
        "español (es)": "es_core_news_md",
        "português (pt)": "pt_core_news_md",
    }
    map_sm = {
        "english (en)": "en_core_web_sm",
        "español (es)": "es_core_news_sm",
        "português (pt)": "pt_core_news_sm",
    }
    target = map_md.get(lang_choice, "xx_ent_wiki_sm")
    fallback = map_sm.get(lang_choice, None)

    def _load(name):
        try: return spacy.load(name)
        except OSError: pass
        try:
            import subprocess, sys
            # Solo intentamos descargar si es estrictamente necesario y permitido
            subprocess.run([sys.executable, "-m", "spacy", "download", name], check=True)
            return spacy.load(name)
        except Exception: return None

    nlp = _load(target)
    if nlp: return nlp
    if fallback and fallback != target:
        nlp = _load(fallback)
        if nlp: return nlp
    return None

def embed_spacy(texts: List[str], nlp) -> np.ndarray:
    vecs = [nlp(t).vector for t in texts]
    return np.vstack(vecs)

@st.cache_resource
def get_sbert_model():
    STClass = _lazy_import_sbert()
    if not STClass: return None
    return STClass('all-MiniLM-L6-v2')

def embed_sbert(texts: List[str]) -> Optional[np.ndarray]:
    model = get_sbert_model()
    if not model: return None
    return model.encode(texts, show_progress_bar=False)

def embed_tfidf(texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    return vec.fit_transform(texts), vec

# ------------------------
# Clustering
# ------------------------
def kmeans_cluster(X, k: int, random_state: int = 42) -> np.ndarray:
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    return model.fit_predict(X)

def try_auto_k(X, k_min=2, k_max=12) -> int:
    best_k, best_score = 5, -1
    n_samples = X.shape[0]
    max_k_limit = min(k_max, max(3, n_samples // 5))
    ks = list(range(k_min, max_k_limit + 1))

    for k in ks:
        try:
            labels = kmeans_cluster(X, k)
            X_data = X.toarray() if hasattr(X, "toarray") else X
            score = silhouette_score(X_data, labels, metric="cosine")
            if score > best_score:
                best_k, best_score = k, score
        except Exception: continue
    return best_k

# ------------------------
# Nombrado con Gemini (SEO Expert Mode - EEAT)
# ------------------------
def name_clusters_seo_mode(df: pd.DataFrame, api_key: str, lang: str) -> pd.DataFrame:
    genai = _lazy_import_gemini()
    if not genai: return df

    try:
        client = genai.Client(api_key=api_key)
    except Exception: return df

    sys_lang = {"Auto": "en", "English": "en", "Español": "es", "Português": "pt"}.get(lang.split(" ")[0], "en")
    
    # Prompt EEAT optimizado
    seo_prompts = {
        "es": """Actúa como un Experto SEO Senior (EEAT).
        Analiza este grupo de keywords.
        Objetivo: Definir un 'Pilar de Contenido' de autoridad.

        Keywords: {keywords}

        Responde SOLO JSON:
        1. "cluster_name": Título de Pilar optimizado (máx 6 palabras).
        2. "user_intent": Intención exacta (ej: 'Informacional - Tutorial', 'Transaccional - Compra').
        3. "content_angle": Frase corta con el enfoque único para ganar autoridad.
        """,
        "en": """Act as a Senior SEO Expert (EEAT).
        Analyze this keyword cluster.
        Goal: Define a high-authority 'Content Pillar'.

        Keywords: {keywords}

        Return ONLY JSON:
        1. "cluster_name": Optimized Pillar Title (max 6 words).
        2. "user_intent": Specific intent (e.g., 'Informational - How-to', 'Transactional - Buy').
        3. "content_angle": Short sentence on the unique authority angle.
        """
    }
    
    prompt_tmpl = seo_prompts.get(sys_lang, seo_prompts["en"])
    
    gen_config = {
        "response_mime_type": "application/json",
        "temperature": 0.2,
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "cluster_name": {"type": "STRING"},
                "user_intent": {"type": "STRING"},
                "content_angle": {"type": "STRING"}
            },
            "required": ["cluster_name", "user_intent", "content_angle"]
        }
    }

    results = []
    groups = list(df.groupby("cluster_id"))
    total_gr = len(groups)
    
    # Progress bar específico para Gemini dentro del status container principal
    prog_text = "🧠 Analizando intención con Gemini..."
    my_bar = st.progress(0, text=prog_text)

    for i, (cid, group) in enumerate(groups):
        my_bar.progress((i+1)/total_gr, text=f"{prog_text} ({i+1}/{total_gr})")
        if i > 0: time.sleep(1) # Rate limit preventivo

        kws = group["keyword_original"].head(15).tolist()
        prompt = prompt_tmpl.format(keywords=", ".join(kws))
        
        # Default
        c_vals = (f"Cluster {cid}", "N/A", "N/A")

        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt],
                    config=gen_config
                )
                d = json.loads(resp.text)
                c_vals = (d.get("cluster_name", c_vals[0]), d.get("user_intent", ""), d.get("content_angle", ""))
                break
            except Exception as e:
                if "429" in str(e) and attempt == 0:
                    time.sleep(4)
                    continue
                else: break
        
        results.append((cid,) + c_vals)

    my_bar.empty()
    res_df = pd.DataFrame(results, columns=["cluster_id", "cluster_name", "user_intent", "content_angle"])
    return df.merge(res_df, on="cluster_id", how="left")

# ==============================================================================
# INTERFAZ DE USUARIO PRINCIPAL (UI)
# ==============================================================================

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.markdown("### 1. Motor de Embeddings")
    embed_method = st.radio(
        "Tecnología",
        ["SBERT (Recomendado)", "spaCy (Rápido)", "TF-IDF (Básico)"],
        index=0,
        help="SBERT ofrece la mejor comprensión semántica para agrupar intenciones de búsqueda."
    )
    
    lang_choice = "auto-multi (xx)"
    if embed_method == "spaCy (Rápido)":
        lang_choice = st.selectbox("Idioma spaCy", ["auto-multi (xx)", "english (en)", "español (es)", "português (pt)"])

    st.markdown("---")
    st.markdown("### 2. Parámetros Clustering")
    auto_k = st.toggle("Detectar K Automático", value=True)
    k_val = st.slider("Número de Clústeres (K)", 2, 50, 10, disabled=auto_k)

    st.markdown("---")
    st.markdown("### 3. Inteligencia Artificial (EEAT)")
    use_gemini = st.toggle("Activar Gemini AI Analysis", value=False)
    gemini_key = ""
    if use_gemini:
        gemini_key = st.text_input("Gemini API Key", type="password", help="Necesaria para generar la estrategia de contenido.")
        if not gemini_key:
            st.warning("⚠️ Se requiere API Key")

# --- ÁREA PRINCIPAL ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("SEO Content Clusterizer")
    st.caption("Agrupación semántica avanzada & Estrategia EEAT con GenAI")

with col_head2:
    # Espacio para branding o estado
    st.write("")

st.markdown("---")

# --- FILE UPLOADER & PREVIEW ---
uploaded_file = st.file_uploader("Sube tu archivo de keywords (CSV)", type=["csv"])

if uploaded_file:
    # 1. Previsualización rápida
    try:
        df_raw = pd.read_csv(uploaded_file, encoding="utf-8-sig", on_bad_lines="skip", dtype=str)
    except:
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, encoding="latin-1", on_bad_lines="skip", dtype=str)

    # Detectar columna
    pos = [c for c in df_raw.columns if c.lower() in {"keyword", "keywords", "query", "kw"}]
    target_col = pos[0] if pos else df_raw.columns[0]

    # Panel de Info y Acción
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric("Keywords detectadas", len(df_raw))
    with c2:
        st.metric("Columna Objetivo", target_col)
    with c3:
        st.write("")
        btn_run = st.button("🚀 Ejecutar Análisis SEO", type="primary", use_container_width=True)

    # --- LÓGICA DE EJECUCIÓN ---
    if btn_run:
        start_time = time.time()
        
        # Contenedor de estado expandible
        with st.status("Procesando tu estrategia de contenidos...", expanded=True) as status:
            
            # PASO 1: Normalización
            st.write("🧹 Limpiando y normalizando keywords...")
            df = df_raw.fillna("")
            df["keyword_original"] = df[target_col].astype("string")
            df["keyword_norm"] = normalize_series(df[target_col])
            df = df[df["keyword_norm"].str.len() > 0].drop_duplicates(subset=["keyword_norm"]).reset_index(drop=True)
            
            # PASO 2: Embeddings
            st.write(f"🧬 Generando embeddings semánticos con {embed_method}...")
            texts = df["keyword_norm"].tolist()
            X = None
            
            if "SBERT" in embed_method:
                X = embed_sbert(texts)
                if X is None:
                    st.warning("⚠️ SBERT no disponible, usando TF-IDF.")
                    X, _ = embed_tfidf(texts)
            elif "spaCy" in embed_method:
                nlp = build_spacy_pipeline(lang_choice)
                if nlp:
                    X = embed_spacy(texts, nlp)
                    if np.allclose(X, 0): X, _ = embed_tfidf(texts)
                else:
                    X, _ = embed_tfidf(texts)
            else:
                X, _ = embed_tfidf(texts)

            # PASO 3: Clustering
            st.write("🧩 Agrupando temas (Clustering)...")
            if auto_k:
                limit = min(20, max(3, len(df)//3))
                final_k = try_auto_k(X, k_max=limit)
                st.info(f"K óptimo detectado: {final_k} clusters")
            else:
                final_k = k_val
            
            df["cluster_id"] = kmeans_cluster(X, final_k)
            
            # PASO 4: Gemini SEO
            if use_gemini and gemini_key:
                st.write("✨ Consultando a Gemini para estrategia EEAT...")
                df = name_clusters_seo_mode(df, gemini_key, lang_choice)
            else:
                df["cluster_name"] = "Cluster " + df["cluster_id"].astype(str)
                df["user_intent"] = "-"
                df["content_angle"] = "-"
            
            status.update(label="¡Análisis completado exitosamente!", state="complete", expanded=False)

        # --- RESULTADOS (TABS) ---
        st.divider()
        st.markdown("### 📊 Estrategia Generada")
        
        tab_strategy, tab_viz, tab_stats = st.tabs(["📝 Estrategia de Contenidos", "🗺️ Mapa Semántico", "📈 Estadísticas"])

        with tab_strategy:
            # Dataframe Principal formateado para lectura SEO
            cols_show = ["cluster_name", "user_intent", "content_angle", "keyword_original"]
            final_df = df.sort_values(["cluster_id"])[cols_show]
            
            st.dataframe(
                final_df,
                column_config={
                    "cluster_name": st.column_config.TextColumn("Pilar de Contenido", help="Tema principal del grupo"),
                    "user_intent": st.column_config.TextColumn("Intención de Usuario", width="medium"),
                    "content_angle": st.column_config.TextColumn("Ángulo de Autoridad (EEAT)", width="large"),
                    "keyword_original": st.column_config.ListColumn("Keywords Agrupadas")
                },
                use_container_width=True,
                height=500
            )
            
            # Botón Descarga
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar Estrategia Completa (CSV)",
                csv,
                "seo_content_strategy.csv",
                "text/csv",
                type="primary"
            )

        with tab_viz:
            st.markdown("#### Visualización de Proximidad (PCA)")
            try:
                X_dense = X.toarray() if hasattr(X, "toarray") else X
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(X_dense)
                viz_df = pd.DataFrame({
                    "x": coords[:,0], "y": coords[:,1],
                    "Pilar": df["cluster_name"],
                    "Keyword": df["keyword_original"]
                })
                st.scatter_chart(viz_df, x="x", y="y", color="Pilar", height=500, use_container_width=True)
            except Exception:
                st.warning("No hay suficientes datos para visualizar el mapa 2D.")

        with tab_stats:
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.markdown("**Distribución de Keywords por Pilar**")
                counts = df["cluster_name"].value_counts()
                st.bar_chart(counts)
            with c_s2:
                st.markdown("**Métricas Globales**")
                st.metric("Total Clusters", final_k)
                st.metric("Promedio Keywords/Cluster", f"{len(df)/final_k:.1f}")

else:
    # Empty State (UX)
    st.markdown("""
    <div style='text-align: center; padding: 4rem; color: #666;'>
        <h3>👋 Bienvenido al SEO Clusterizer</h3>
        <p>Sube tu archivo CSV para comenzar a detectar oportunidades de contenido.</p>
    </div>
    """, unsafe_allow_html=True)
