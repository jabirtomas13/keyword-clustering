#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword Clustering App (ES/EN/PT) – UI Refactorizada
------------------------------------------------
Integración de lógica de clustering existente con nueva interfaz UX/UI.
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
    page_title="Keyword Cluster Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stButton>button { border-radius: 20px; font-weight: 600; width: 100%; transition: all 0.3s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    h1 { color: #1E1E1E; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Imports perezosos (Lazy Imports) - LÓGICA ORIGINAL
# ------------------------
def _lazy_import_spacy():
    try:
        import spacy
        return spacy
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
# Utilidades de Normalización - LÓGICA ORIGINAL
# ------------------------
_SMARTS = {
    "\u2018": "'", "\u2019": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u00A0": " ",
}
_SMARTS_TRANS = str.maketrans(_SMARTS)
_EMOJI_RE = re.compile("[\\U0001F300-\\U0001F6FF\\U0001F700-\\U0001FAFF\\U00002700-\\U000027BF\\U0001F900-\\U0001F9FF]")
_ALLOWED_RE = re.compile(r"[^a-z0-9\-\' ]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

def normalize_keyword(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    if not s: return ""
    s = s.translate(_SMARTS_TRANS)
    s = _EMOJI_RE.sub(" ", s)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.lower()
    s = _ALLOWED_RE.sub(" ", s)
    s = re.sub(r"\b[-']+|[-']+\b", " ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s

def normalize_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").map(normalize_keyword)

# ------------------------
# Embeddings y Clustering - LÓGICA ORIGINAL
# ------------------------
@st.cache_resource
def build_spacy_pipeline(lang_choice: str):
    spacy_lib = _lazy_import_spacy()
    if not spacy_lib: return None
    
    # Mapas simplificados para la demo
    map_model = {
        "english (en)": "en_core_web_sm",
        "español (es)": "es_core_news_sm",
        "português (pt)": "pt_core_news_sm",
        "auto-multi (xx)": "xx_ent_wiki_sm"
    }
    target = map_model.get(lang_choice, "en_core_web_sm")
    
    try:
        return spacy_lib.load(target)
    except OSError:
        try:
            # Fallback a descarga silenciosa si es posible
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", target], check=True)
            return spacy_lib.load(target)
        except:
            return None

def embed_spacy(texts: List[str], nlp) -> np.ndarray:
    vecs = [nlp(t).vector for t in texts]
    return np.vstack(vecs)

def embed_tfidf(texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    return vec.fit_transform(texts), vec

def kmeans_cluster(X, k: int, random_state: int = 42) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)

def try_auto_k(X, k_min=2, k_max=12) -> int:
    best_k, best_score = 5, -1
    ks = list(range(k_min, min(k_max, max(3, X.shape[0] // 5)) + 1))
    for k in ks:
        try:
            score = silhouette_score(X, kmeans_cluster(X, k), metric="cosine")
            if score > best_score: best_k, best_score = k, score
        except: continue
    return best_k

# ------------------------
# Gemini Integration - LÓGICA ORIGINAL
# ------------------------
def name_clusters_with_gemini(df: pd.DataFrame, api_key: str, lang: str) -> pd.DataFrame:
    genai = _lazy_import_gemini()
    if not genai: return df
    
    try:
        client = genai.Client(api_key=api_key)
    except: return df

    prompts = {
        "en": "Given keywords: {keywords}. Return JSON with 'cluster_name' (max 5 words) and 'description'.",
        "es": "Dadas las palabras clave: {keywords}. Devuelve JSON con 'cluster_name' (máx 5 palabras) y 'description'.",
        "pt": "Dadas as palavras-chave: {keywords}. Retorne JSON com 'cluster_name' (máx 5 palavras) e 'description'."
    }
    sys_lang = {"Auto": "en", "English": "en", "Español": "es", "Português": "pt"}.get(lang.split(" ")[0], "en")
    
    results = []
    groups = list(df.groupby("cluster_id"))
    
    progress_bar = st.progress(0, text="Consultando a Gemini...")
    
    for i, (cid, group) in enumerate(groups):
        progress_bar.progress((i+1)/len(groups), text=f"Nombrando clúster {cid}...")
        if i > 0: time.sleep(2) # Rate limit simple
        
        kws = group["keyword_original"].head(15).tolist()
        prompt = prompts.get(sys_lang, prompts["en"]).format(keywords=", ".join(kws))
        
        try:
            resp = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": {"type": "OBJECT", "properties": {"cluster_name": {"type": "STRING"}, "description": {"type": "STRING"}}}
                }
            )
            data = json.loads(resp.text)
            results.append((cid, data.get("cluster_name", f"Cluster {cid}"), data.get("description", "")))
        except:
            results.append((cid, f"Cluster {cid}", ""))
            
    progress_bar.empty()
    return df.merge(pd.DataFrame(results, columns=["cluster_id", "cluster_name", "cluster_description"]), on="cluster_id", how="left")

# ==============================================================================
# NUEVA INTERFAZ DE USUARIO (UI)
# ==============================================================================

# --- BARRA LATERAL (CONFIGURACIÓN) ---
with st.sidebar:
    st.title("⚙️ Configuración")
    
    st.markdown("### 1. Parámetros de Modelo")
    lang_choice = st.selectbox(
        "Idioma del Texto",
        ["auto-multi (xx)", "english (en)", "español (es)", "português (pt)"],
        index=2,
        help="Selecciona el idioma principal de tus keywords."
    )
    
    embed_method = st.radio(
        "Motor de Embeddings", 
        ["TF-IDF (Rápido & Simple)", "spaCy vectors (Semántico)"], 
        index=0,
        help="TF-IDF agrupa por palabras compartidas. SpaCy entiende significados similares."
    )
    
    st.markdown("---")
    st.markdown("### 2. Clustering")
    auto_k = st.toggle("Detectar K automáticamente", value=True)
    k_clusters = st.slider("Número de Clústeres (K)", 2, 50, 8, disabled=auto_k)
    do_dimred = st.toggle("Generar Gráfico PCA", value=True)
    
    st.markdown("---")
    st.markdown("### 3. Inteligencia Artificial")
    use_gemini = st.toggle("Nombrar con Google Gemini", value=False)
    
    gemini_key = ""
    if use_gemini:
        gemini_key = st.text_input("Gemini API Key", type="password", help="Necesaria para generar nombres automáticamente.")
        if not gemini_key:
            st.warning("⚠️ Ingresa tu API Key para activar esta función.")

    st.markdown("---")
    st.info("💡 **Tip:** Asegúrate de que tu archivo CSV tenga una columna llamada 'keyword', 'query' o 'kw'.")

# --- ÁREA PRINCIPAL ---
col_header_1, col_header_2 = st.columns([3, 1])
with col_header_1:
    st.title("Keyword Cluster Studio")
    st.markdown("Sube tus palabras clave y utiliza Inteligencia Artificial para agruparlas semánticamente.")
with col_header_2:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60) # Icono decorativo

st.markdown("---")

# --- SECCIÓN DE ENTRADA ---
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"], help="Arrastra tu archivo aquí")

if uploaded_file:
    # Pre-lectura para mostrar info rápida
    try:
        df_raw = pd.read_csv(uploaded_file, encoding="utf-8-sig", on_bad_lines="skip", dtype=str)
    except:
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file, encoding="latin-1", on_bad_lines="skip", dtype=str)
    
    col_info1, col_info2, col_btn = st.columns([2, 2, 2])
    with col_info1:
        st.metric("Filas detectadas", len(df_raw))
    with col_info2:
        # Detectar columna
        possible = [c for c in df_raw.columns if c.lower() in {"keyword", "keywords", "query", "kw"}]
        target_col = possible[0] if possible else df_raw.columns[0]
        st.metric("Columna Objetivo", target_col)
    
    with col_btn:
        st.write("") # Espaciador
        btn_process = st.button("🚀 Ejecutar Clustering", type="primary")

    # --- LÓGICA DE PROCESAMIENTO (Solo si se pulsa el botón) ---
    if btn_process:
        with st.status("Procesando datos...", expanded=True) as status:
            
            # 1. Normalización
            st.write("🔄 Normalizando texto...")
            df = df_raw.fillna("")
            df["keyword_original"] = df[target_col].astype("string")
            df["keyword_norm"] = normalize_series(df[target_col])
            df = df[df["keyword_norm"].str.len() > 0].drop_duplicates(subset=["keyword_norm"]).reset_index(drop=True)
            
            # 2. Embeddings
            st.write(f"🧮 Calculando vectores usando {embed_method}...")
            texts = df["keyword_norm"].tolist()
            if "spaCy" in embed_method:
                nlp = build_spacy_pipeline(lang_choice)
                if nlp:
                    X = embed_spacy(texts, nlp)
                    if np.allclose(X, 0): 
                        st.warning("⚠️ Vectores nulos, cambiando a TF-IDF.")
                        X, _ = embed_tfidf(texts)
                else:
                    X, _ = embed_tfidf(texts)
            else:
                X, _ = embed_tfidf(texts)
            
            # 3. Clustering
            if auto_k:
                st.write("🤖 Calculando K óptimo...")
                final_k = try_auto_k(X, k_max=min(15, len(df)//2))
            else:
                final_k = k_clusters
            
            st.write(f"🧩 Agrupando en {final_k} clústeres...")
            df["cluster_id"] = kmeans_cluster(X, final_k)
            
            # 4. Gemini (Opcional)
            if use_gemini and gemini_key:
                st.write("✨ Generando nombres con Gemini AI...")
                df = name_clusters_with_gemini(df, gemini_key, lang_choice)
            else:
                df["cluster_name"] = "Cluster " + df["cluster_id"].astype(str)
                df["cluster_description"] = "-"

            status.update(label="¡Proceso completado!", state="complete", expanded=False)

        # --- RESULTADOS VISUALES ---
        st.markdown("### 📊 Resultados del Análisis")
        
        tab_viz, tab_data, tab_stats = st.tabs(["🗺️ Visualización", "📄 Datos Detallados", "📈 Estadísticas"])
        
        with tab_viz:
            if do_dimred:
                try:
                    X_dense = X.toarray() if hasattr(X, "toarray") else X
                    pca = PCA(n_components=2, random_state=42)
                    coords = pca.fit_transform(X_dense)
                    viz_df = pd.DataFrame({
                        "x": coords[:,0], "y": coords[:,1], 
                        "Cluster": df["cluster_name"], 
                        "Keyword": df["keyword_original"]
                    })
                    st.scatter_chart(viz_df, x="x", y="y", color="Cluster", height=500)
                except Exception as e:
                    st.error(f"No se pudo generar el gráfico PCA: {e}")
            else:
                st.info("La visualización PCA está desactivada en la configuración.")

        with tab_data:
            # Preparar DF limpio para mostrar/descargar
            final_cols = ["cluster_id", "cluster_name", "keyword_original", "cluster_description"]
            other_cols = [c for c in df.columns if c not in final_cols and c != "keyword_norm"]
            df_display = df[final_cols + other_cols].sort_values("cluster_id")
            
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            csv = df_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar Resultados (CSV)",
                csv,
                "keywords_clustered.csv",
                "text/csv",
                type="primary"
            )

        with tab_stats:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("#### Distribución por Clúster")
                counts = df["cluster_name"].value_counts()
                st.bar_chart(counts)
            with col_s2:
                st.markdown("#### Métricas Rápidas")
                st.metric("Total Keywords Únicas", len(df))
                st.metric("Total Clústeres", final_k)
                avg_size = len(df) / final_k
                st.metric("Tamaño Promedio", f"{avg_size:.1f} kws")

else:
    # Estado vacío (Placeholder)
    st.markdown("""
    <div style='text-align: center; padding: 50px; color: #888;'>
        <h3>👋 Bienvenido al Studio</h3>
        <p>Sube un archivo CSV en el panel superior para comenzar.</p>
    </div>
    """, unsafe_allow_html=True)
