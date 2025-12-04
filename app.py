#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword Clustering App (ES/EN/PT) – GitHub-ready
------------------------------------------------
- Carga robusta de CSV
- Normalización multilenguaje (es/en/pt)
- Clustering KMeans con detección de K (Silhouette)
- Embeddings opcionales con spaCy (fallback a TF-IDF)
- Nombres de clústeres con Google Gemini (Integración corregida)
"""
import os
import re
import unicodedata
import json
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

# Rango básico de emojis/pictogramas
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F6FF\U0001F700-\U0001FAFF\U00002700-\U000027BF\U0001F900-\U0001F9FF]"
)
# Mantener letras/números/espacios, guiones, apóstrofes
_ALLOWED_RE = re.compile(r"[^a-z0-9\-\' ]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

def strip_diacritics(text: str) -> str:
    """Elimina tildes y diacríticos del texto."""
    norm = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in norm if not unicodedata.combining(ch))

def normalize_keyword(s: str) -> str:
    """Realiza una normalización robusta de la palabra clave."""
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
    """Aplica normalización a una Serie de pandas."""
    return series.astype("string").fillna("").map(normalize_keyword)

# ------------------------
# Embeddings (Vectores)
# ------------------------
def build_spacy_pipeline(lang_choice: str):
    """Carga o descarga el modelo de spaCy especificado."""
    spacy = _lazy_import_spacy()
    if not spacy:
        return None

    model_map = {
        "auto-multi (xx)": "xx_ent_wiki_sm",
        "english (en)": "en_core_web_md",
        "español (es)": "es_core_news_md",
        "português (pt)": "pt_core_news_md",
    }
    model_name = model_map.get(lang_choice, "xx_ent_wiki_sm")
    try:
        return spacy.load(model_name)
    except Exception:
        # Intentar descargar al vuelo
        try:
            import subprocess, sys
            st.info(f"Descargando modelo spaCy '{model_name}'...")
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
            return spacy.load(model_name)
        except Exception:
            return None

def embed_spacy(texts: List[str], nlp) -> np.ndarray:
    """Genera vectores usando spaCy."""
    vecs = []
    for t in texts:
        doc = nlp(t)
        vecs.append(doc.vector)
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
    """Ejecuta K-Means."""
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return labels

def try_auto_k(X, k_min=2, k_max=12) -> int:
    """
    Encuentra el K óptimo usando Silhouette Score.
    CORREGIDO: Usa X.shape[0] para compatibilidad con matrices dispersas.
    """
    best_k, best_score = None, -1

    # --- CORRECCIÓN APLICADA AQUÍ ---
    n_samples = X.shape[0] 
    # --------------------------------

    # Determinar límite de K basado en el tamaño de datos
    max_k_limit = min(k_max, max(3, n_samples // 5))

    ks = list(range(k_min, max_k_limit + 1))

    for k in ks:
        try:
            labels = kmeans_cluster(X, k)
            # Usar X directo si es denso, o tal cual si es disperso (silhouette lo maneja si se pasa metric)
            X_data = X
            # Usamos cosine distance para texto
            score = silhouette_score(X_data, labels, metric="cosine")
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue
    return best_k or 5

# ------------------------
# Nombrado con Gemini (Opcional)
# ------------------------
def name_clusters_with_gemini(df: pd.DataFrame, api_key: str, lang: str) -> pd.DataFrame:
    """Usa la API de Gemini para nombrar clústeres."""
    genai = _lazy_import_gemini()
    if not genai:
        st.warning("Paquete google-genai no disponible. Omite nombres de clúster.")
        return df

    try:
        # Inicializar cliente Gemini
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error al inicializar cliente Gemini: {e}")
        return df

    prompts = {
        "en": """Given the following keywords, identify a common theme and return:
1) A short cluster name (max. 5 words)
2) A one-sentence description.
Keywords: {keywords}
Respond strictly in the requested **JSON** format.
""",
        "es": """Dadas las siguientes palabras clave, identifica un tema común y devuelve:
1) Un nombre corto para el grupo (máx. 5 palabras)
2) Una breve descripción en una oración.
Palabras clave: {keywords}
Responde estrictamente en el formato **JSON** solicitado.
""",
        "pt": """Dadas as palavras-chave a seguir, identifique um tema comum e retorne:
1) Um nome curto para o grupo (máx. 5 palavras)
2) Uma breve descrição em uma frase.
Palavras-chave: {keywords}
Responda estritamente no formato **JSON** solicitado.
""",
    }

    # Heurística simple de idioma basada en la selección de UI
    sys_lang = {"Auto": "en", "English": "en", "Español": "es", "Português": "pt"}.get(lang.split(" ")[0], "en")

    # Instrucción del sistema
    system_instruction = f"""Actúa como un analista de datos experto. Tu única tarea es nombrar un grupo de palabras clave.
    Siempre debes responder en JSON. La estructura JSON requerida es:
    {{
      "cluster_name": "Nombre descriptivo",
      "description": "Breve explicación de la categoría."
    }}
    La respuesta debe ser en idioma '{sys_lang}'."""

    # --- CORRECCIÓN APLICADA AQUÍ (Configuración Gemini) ---
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.2,
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "cluster_name": {"type": "STRING", "description": "Short, descriptive name for the cluster (max 5 words)."},
                "description": {"type": "STRING", "description": "One-sentence explanation of the cluster theme."}
            },
            "required": ["cluster_name", "description"]
        },
        # La instrucción del sistema va DENTRO de config en el nuevo SDK
        "system_instruction": system_instruction
    }
    # -------------------------------------------------------

    results = []
    
    # Procesar clústeres
    for cl_id, group in df.groupby("cluster_id"):
        kws = group["keyword_original"].head(15).tolist()
        kw_blob = ", ".join(kws)
        prompt = prompts[sys_lang].format(keywords=kw_blob)

        try:
            # Llamada a Gemini
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt],
                config=generation_config
                # NOTA: system_instruction ya no se pasa aquí
            )

            content = response.text
            data = json.loads(content)

            cluster_name = data.get("cluster_name") or f"Cluster {cl_id}"
            description = data.get("description") or ""

        except Exception as e:
            st.error(f"Error al nombrar el Clúster {cl_id} con Gemini: {e}")
            cluster_name, description = f"Cluster {cl_id}", "Error al generar la descripción."

        results.append((cl_id, cluster_name, description))

    name_df = pd.DataFrame(results, columns=["cluster_id", "cluster_name", "cluster_description"])
    return df.merge(name_df, on="cluster_id", how="left")

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
    st.info("💡 Consejo: la columna debe llamarse `keyword`, `keywords`, `query` o `kw` (o será la primera columna).")
    st.stop()

# Lectura robusta del CSV
try:
    df = pd.read_csv(uploaded, encoding="utf-8-sig", on_bad_lines="skip", dtype=str)
except Exception:
    uploaded.seek(0)
    df = pd.read_csv(uploaded, encoding="latin-1", on_bad_lines="skip", dtype=str)

df = df.fillna("")
possible = [c for c in df.columns if c.lower() in {"keyword", "keywords", "query", "kw"}]
kw_col = possible[0] if possible else df.columns[0]

# Crear columnas original y normalizada
df["keyword_original"] = df[kw_col].astype("string")
df["keyword_norm"] = normalize_series(df[kw_col])
df = df[df["keyword_norm"].str.len() > 0].drop_duplicates(subset=["keyword_norm"]).reset_index(drop=True)

st.success(f"✅ CSV cargado. Usando columna: **{kw_col}** · Filas: **{len(df)}**")
with st.expander("Ver muestra de normalización", expanded=False):
    st.dataframe(df[["keyword_original", "keyword_norm"]].head(20))

# Construir Vectores
texts = df["keyword_norm"].tolist()

X = None
vectorizer = None
nlp = None

if embed_method == "spaCy vectors":
    nlp = build_spacy_pipeline(lang_choice)
    if nlp is not None and nlp.vocab.vectors.shape[0] > 0:
        with st.spinner("Calculando vectores de spaCy..."):
            X = embed_spacy(texts, nlp)
            # Algunos modelos pequeños devuelven vectores ceros
            if np.allclose(X, 0):
                st.warning("Los vectores de este modelo parecen nulos. Cambio a TF-IDF.")
                X, vectorizer = embed_tfidf(texts)
    else:
        st.warning("No se pudo cargar un modelo de spaCy válido. Uso TF-IDF.")
        X, vectorizer = embed_tfidf(texts)
else:
    X, vectorizer = embed_tfidf(texts)

# Decidir K
if auto_k:
    with st.spinner("Buscando K óptimo (silhouette)..."):
        # try_auto_k ya incluye el fix de len(X) -> X.shape[0]
        k_auto = try_auto_k(X, k_min=2, k_max=min(12, max(3, len(df)//5)))
        if k_auto:
            k = k_auto
st.write(f"**K seleccionado:** {k}")

# Clustering
labels = kmeans_cluster(X, k)
df["cluster_id"] = labels

# Visualización PCA Opcional
if do_dimred:
    st.markdown("### 2) Visualización 2D (PCA)")
    try:
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = X
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_dense)
        viz = pd.DataFrame({"x": coords[:,0], "y": coords[:,1], "cluster": df["cluster_id"], "keyword": df["keyword_original"]})
        st.scatter_chart(viz, x="x", y="y", color="cluster", size=None)
    except Exception as e:
        st.warning(f"No se pudo proyectar a 2D: {e}")

# Nombrado con Gemini
if use_gemini and gemini_key:
    with st.spinner("Nombrando clústeres con Gemini..."):
        df = name_clusters_with_gemini(df, api_key=gemini_key, lang=lang_choice)
else:
    # Inicializar vacíos si no se usa Gemini
    df["cluster_name"] = ""
    df["cluster_description"] = ""

# --- 🛡️ CORRECCIÓN APLICADA AQUÍ (BLINDAJE) ---
# Si Gemini falló o devolvió el DF original, asegurarse de que las columnas existan
if "cluster_name" not in df.columns:
    df["cluster_name"] = "Sin nombre (Gemini omitido)"
if "cluster_description" not in df.columns:
    df["cluster_description"] = ""
# ----------------------------------------------

# Output Final
st.markdown("### 3) Resultado")
order_cols = ["keyword_original", "keyword_norm", "cluster_id", "cluster_name", "cluster_description"]
extra_cols = [c for c in df.columns if c not in order_cols]
# Ahora esto es seguro porque las columnas están garantizadas
df = df[order_cols + extra_cols]

st.dataframe(df)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV clusterizado", data=csv_bytes, file_name="clustered_keywords.csv", mime="text/csv")

st.caption("Hecho para para es/en/pt · Normalización robusta · Gemini API")
