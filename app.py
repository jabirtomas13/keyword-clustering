#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword Clustering App (ES/EN/PT) – GitHub-ready
------------------------------------------------
- Carga robusta de CSV
- Normalización multilenguaje (es/en/pt)
- Clustering KMeans con detección de K (Silhouette)
- Embeddings opcionales con spaCy (fallback inteligente md -> sm -> TF-IDF)
- Nombres de clústeres con Google Gemini (Integración corregida + Rate Limit Fix)
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

# ------------------------
# Imports perezosos (Lazy Imports)
# ------------------------
def _lazy_import_spacy():
    """Intenta importar la librería spaCy."""
    try:
        import spacy
        return spacy
    except Exception:
        return None

def _lazy_import_gemini():
    """Intenta importar la librería google-genai."""
    try:
        from google import genai
        # Verificar que tenga el cliente
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
    "\u00A0": " ",  # espacio sin ruptura
}
_SMARTS_TRANS = str.maketrans(_SMARTS)

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F6FF\U0001F700-\U0001FAFF\U00002700-\U000027BF\U0001F900-\U0001F9FF]"
)
_ALLOWED_RE = re.compile(r"[^a-z0-9\-\' ]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

def strip_diacritics(text: str) -> str:
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))

def normalize_keyword(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
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
# Embeddings (Vectores) con Fallback Robusto
# ------------------------
def build_spacy_pipeline(lang_choice: str):
    """
    Carga modelo spaCy con estrategia de fallback:
    1. Intenta cargar modelo MD (Medium).
    2. Si falla (error descarga/memoria), intenta descargar y cargar SM (Small).
    3. Si falla, retorna None (para usar TF-IDF).
    """
    spacy = _lazy_import_spacy()
    if not spacy:
        return None

    # Mapas de modelos
    map_md = {
        "auto-multi (xx)": "xx_ent_wiki_sm", # xx suele ser sm
        "english (en)": "en_core_web_md",
        "español (es)": "es_core_news_md",
        "português (pt)": "pt_core_news_md",
    }
    
    # Mapas de fallback (modelos pequeños)
    map_sm = {
        "english (en)": "en_core_web_sm",
        "español (es)": "es_core_news_sm",
        "português (pt)": "pt_core_news_sm",
    }

    target_model = map_md.get(lang_choice, "xx_ent_wiki_sm")
    fallback_model = map_sm.get(lang_choice, None)

    def _try_load_or_download(model_name):
        # 1. Intentar cargar si ya existe
        try:
            return spacy.load(model_name)
        except OSError:
            pass # No instalado
        
        # 2. Intentar descargar
        try:
            import subprocess, sys
            st.info(f"⬇️ Descargando modelo '{model_name}'... (esto puede tardar)")
            # Usamos subprocess.run. Si falla por permisos (exit 1), saltará al except
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
            return spacy.load(model_name)
        except Exception as e:
            # Capturamos el error silenciosamente aquí para intentar el fallback fuera
            return None

    # A) Intentar modelo principal (MD)
    nlp_obj = _try_load_or_download(target_model)
    if nlp_obj:
        return nlp_obj

    # B) Si falló el principal y existe uno más ligero, probar ese
    if fallback_model and fallback_model != target_model:
        st.warning(f"⚠️ No se pudo cargar '{target_model}' (posible error de permisos/memoria). Intentando versión ligera '{fallback_model}'...")
        nlp_obj = _try_load_or_download(fallback_model)
        if nlp_obj:
            st.success(f"✅ Modelo '{fallback_model}' cargado correctamente.")
            return nlp_obj

    # C) Fallo total
    st.error("❌ No se pudieron descargar los modelos de spaCy (entorno restringido). Se usará TF-IDF automáticamente.")
    return None

def embed_spacy(texts: List[str], nlp) -> np.ndarray:
    """Genera vectores usando spaCy."""
    vecs = []
    # Barra de progreso para embeddings si son muchos datos
    prog_bar = st.progress(0)
    total = len(texts)
    
    for i, t in enumerate(texts):
        if i % 100 == 0:
            prog_bar.progress((i+1)/total)
        doc = nlp(t)
        vecs.append(doc.vector)
    
    prog_bar.empty()
    arr = np.vstack(vecs)
    return arr

def embed_tfidf(texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
    """Genera vectores usando TF-IDF."""
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(texts)
    return X, vec

# ------------------------
# Clustering
# ------------------------
def kmeans_cluster(X, k: int, random_state: int = 42) -> np.ndarray:
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return labels

def try_auto_k(X, k_min=2, k_max=12) -> int:
    best_k, best_score = None, -1
    n_samples = X.shape[0]
    max_k_limit = min(k_max, max(3, n_samples // 5))
    ks = list(range(k_min, max_k_limit + 1))

    for k in ks:
        try:
            labels = kmeans_cluster(X, k)
            score = silhouette_score(X, labels, metric="cosine") # X ya es compatible
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue
    return best_k or 5

# ------------------------
# Nombrado con Gemini
# ------------------------
def name_clusters_with_gemini(df: pd.DataFrame, api_key: str, lang: str) -> pd.DataFrame:
    genai = _lazy_import_gemini()
    if not genai:
        st.warning("Paquete google-genai no disponible. Omite nombres de clúster.")
        return df

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error al inicializar cliente Gemini: {e}")
        return df

    prompts = {
        "en": "Given keywords: {keywords}. Return JSON with 'cluster_name' (max 5 words) and 'description'.",
        "es": "Dadas las palabras clave: {keywords}. Devuelve JSON con 'cluster_name' (máx 5 palabras) y 'description'.",
        "pt": "Dadas as palavras-chave: {keywords}. Retorne JSON com 'cluster_name' (máx 5 palavras) e 'description'.",
    }
    sys_lang = {"Auto": "en", "English": "en", "Español": "es", "Português": "pt"}.get(lang.split(" ")[0], "en")
    
    system_instr = f"Actúa como analista. Responde SOLO JSON válido. Idioma: {sys_lang}."
    
    gen_config = {
        "response_mime_type": "application/json",
        "temperature": 0.2,
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "cluster_name": {"type": "STRING"},
                "description": {"type": "STRING"}
            },
            "required": ["cluster_name", "description"]
        },
        "system_instruction": system_instr
    }

    results = []
    groups = list(df.groupby("cluster_id"))
    total = len(groups)
    
    bar = st.progress(0)
    status = st.empty()

    for i, (cid, group) in enumerate(groups):
        if i > 0: time.sleep(4)
        bar.progress((i+1)/total)
        status.text(f"Nombrando clúster {cid}...")

        kws = group["keyword_original"].head(15).tolist()
        prompt = prompts.get(sys_lang, prompts["en"]).format(keywords=", ".join(kws))

        c_name, c_desc = f"Cluster {cid}", ""
        
        # Reintentos básicos
        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt],
                    config=gen_config
                )
                data = json.loads(resp.text)
                c_name = data.get("cluster_name", c_name)
                c_desc = data.get("description", "")
                break
            except Exception as e:
                if "429" in str(e) and attempt == 0:
                    time.sleep(10)
                    continue
                else:
                    print(f"Err {cid}: {e}")
                    break
        
        results.append((cid, c_name, c_desc))

    bar.empty()
    status.empty()
    return df.merge(pd.DataFrame(results, columns=["cluster_id", "cluster_name", "cluster_description"]), on="cluster_id", how="left")

# ------------------------
# UI Principal
# ------------------------
st.set_page_config(page_title="Keyword Clustering (ES/EN/PT)", layout="wide")

st.title("🔎 Keyword Clustering (ES · EN · PT)")
st.caption("Carga tu CSV, normalizamos los términos y agrupamos por similitud. Ahora impulsado por la API de Gemini.")

with st.sidebar:
    st.header("⚙️ Configuración")
    lang_choice = st.selectbox(
        "Idioma para spaCy (solo si eliges 'spaCy vectors')",
        ["auto-multi (xx)", "english (en)", "español (es)", "português (pt)"],
        index=0,
    )
    embed_method = st.radio("Método de vectores", ["TF-IDF (rápido)", "spaCy vectors"], index=0)
    auto_k = st.checkbox("Elegir K automáticamente (silhouette)", value=True)
    k = st.slider("Número de clústeres (K)", 2, 30, 8)
    do_dimred = st.checkbox("PCA 2D para visualización", value=True)
    use_gemini = st.checkbox("Nombrar clústeres con Gemini", value=False)
    if use_gemini:
        gemini_key = st.text_input("GEMINI_API_KEY", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    else:
        gemini_key = ""

st.markdown("### 1) Sube tu CSV de keywords")
uploaded = st.file_uploader("CSV con columna de keywords", type=["csv"])

if uploaded is None:
    st.info("💡 Consejo: la columna debe llamarse `keyword`, `keywords`, `query` o `kw`.")
    st.stop()

try:
    df = pd.read_csv(uploaded, encoding="utf-8-sig", on_bad_lines="skip", dtype=str)
except Exception:
    uploaded.seek(0)
    df = pd.read_csv(uploaded, encoding="latin-1", on_bad_lines="skip", dtype=str)

df = df.fillna("")
possible = [c for c in df.columns if c.lower() in {"keyword", "keywords", "query", "kw"}]
kw_col = possible[0] if possible else df.columns[0]

df["keyword_original"] = df[kw_col].astype("string")
df["keyword_norm"] = normalize_series(df[kw_col])
df = df[df["keyword_norm"].str.len() > 0].drop_duplicates(subset=["keyword_norm"]).reset_index(drop=True)

st.success(f"✅ CSV cargado. Columna: **{kw_col}** · Filas: **{len(df)}**")

# Construir Vectores
texts = df["keyword_norm"].tolist()
X = None
nlp = None

if embed_method == "spaCy vectors":
    nlp = build_spacy_pipeline(lang_choice)
    if nlp:
        with st.spinner("Calculando vectores de spaCy..."):
            X = embed_spacy(texts, nlp)
            if np.allclose(X, 0):
                st.warning("Vectores nulos detectados. Cambiando a TF-IDF.")
                X, _ = embed_tfidf(texts)
    else:
        # Fallback manejado dentro de build_spacy_pipeline, pero si retorna None final:
        st.info("Usando TF-IDF debido a fallo en carga de modelo.")
        X, _ = embed_tfidf(texts)
else:
    X, _ = embed_tfidf(texts)

# K Automático
if auto_k:
    with st.spinner("Buscando K óptimo..."):
        k_auto = try_auto_k(X, k_min=2, k_max=min(12, max(3, len(df)//5)))
        if k_auto: k = k_auto
st.write(f"**K seleccionado:** {k}")

# Clustering
df["cluster_id"] = kmeans_cluster(X, k)

# Visualización
if do_dimred:
    st.markdown("### 2) Visualización 2D (PCA)")
    try:
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_dense)
        viz = pd.DataFrame({"x": coords[:,0], "y": coords[:,1], "cluster": df["cluster_id"], "keyword": df["keyword_original"]})
        st.scatter_chart(viz, x="x", y="y", color="cluster")
    except Exception as e:
        st.warning(f"Error en PCA: {e}")

# Gemini
if use_gemini and gemini_key:
    df = name_clusters_with_gemini(df, gemini_key, lang_choice)
else:
    df["cluster_name"] = ""
    df["cluster_description"] = ""

if "cluster_name" not in df.columns: df["cluster_name"] = "Sin nombre"
if "cluster_description" not in df.columns: df["cluster_description"] = ""

st.markdown("### 3) Resultado")
final_cols = ["keyword_original", "keyword_norm", "cluster_id", "cluster_name", "cluster_description"]
other = [c for c in df.columns if c not in final_cols]
df = df[final_cols + other]

st.dataframe(df)
st.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8"), "clustered_keywords.csv", "text/csv")
