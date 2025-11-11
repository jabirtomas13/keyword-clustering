#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword Clustering App (ES/EN/PT) – GitHub-ready
------------------------------------------------
- Robust CSV loading
- Cross-language normalization (es/en/pt)
- KMeans clustering
- Optional spaCy embeddings (falls back to TF-IDF)
- Cluster naming via OpenAI (optional)
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

# --- Optional deps (lazy import) ---
def _lazy_import_spacy():
    try:
        import spacy
        return spacy
    except Exception:
        return None

def _lazy_import_openai():
    try:
        from openai import OpenAI
        return OpenAI
    except Exception:
        return None

# ------------------------
# Normalization utilities
# ------------------------
_SMARTS = {
    "\u2018": "'", "\u2019": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u00A0": " ",  # non-breaking space
}
_SMARTS_TRANS = str.maketrans(_SMARTS)

# Basic emoji/pictograph range
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F6FF\U0001F700-\U0001FAFF\U00002700-\U000027BF\U0001F900-\U0001F9FF]"
)
# Keep letters/numbers/spaces, hyphens, apostrophes
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
# Embeddings
# ------------------------
def build_spacy_pipeline(lang_choice: str):
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
        # try to download on the fly
        try:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
            return spacy.load(model_name)
        except Exception:
            return None

def embed_spacy(texts: List[str], nlp) -> np.ndarray:
    vecs = []
    for t in texts:
        doc = nlp(t)
        vecs.append(doc.vector)
    arr = np.vstack(vecs)
    # If zero vectors (small models), fallback to TF-IDF later
    return arr

def embed_tfidf(texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(texts)
    return X, vec

# ------------------------
# Clustering
# ------------------------
def kmeans_cluster(X, k: int, random_state: int = 42) -> np.ndarray:
    if hasattr(X, "toarray"):
        # sparse matrix
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
    else:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X)
    return labels

def try_auto_k(X, k_min=2, k_max=12) -> int:
    best_k, best_score = None, -1
    ks = list(range(k_min, max(k_min+1, k_max+1)))
    for k in ks:
        try:
            labels = kmeans_cluster(X, k)
            score = silhouette_score(X if hasattr(X, "toarray") else X, labels, metric="cosine")
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue
    return best_k or 5

# ------------------------
# OpenAI naming (optional)
# ------------------------
def name_clusters_with_openai(df: pd.DataFrame, api_key: str, lang: str) -> pd.DataFrame:
    OpenAI = _lazy_import_openai()
    if not OpenAI:
        st.warning("Paquete openai no disponible. Omite nombres de clúster.")
        return df
    client = OpenAI(api_key=api_key)

    prompts = {
        "en": """Given the following keywords, identify a common theme and return:
1) A short cluster name (max. 5 words)
2) A one-sentence description.
Keywords: {keywords}
Respond in **JSON** like:
{{
  "cluster_name": "Descriptive name",
  "description": "Brief explanation of the category."
}}""",
        "es": """Dadas las siguientes palabras clave, identifica un tema común y devuelve:
1) Un nombre corto para el grupo (máx. 5 palabras)
2) Una breve descripción en una oración.
Palabras clave: {keywords}
Responde en **JSON** así:
{{
  "cluster_name": "Nombre descriptivo",
  "description": "Breve explicación de la categoría."
}}""",
        "pt": """Dadas as palavras-chave a seguir, identifique um tema comum e retorne:
1) Um nome curto para o grupo (máx. 5 palavras)
2) Uma breve descrição em uma frase.
Palavras-chave: {keywords}
Responda em **JSON** assim:
{{
  "cluster_name": "Nome descritivo",
  "description": "Breve explicação da categoria."
}}""",
    }
    # Simple language heuristic from UI choice
    sys_lang = {"Auto": "en", "English": "en", "Español": "es", "Português": "pt"}.get(lang, "en")
    results = []
    for cl_id, group in df.groupby("cluster_id"):
        kws = group["keyword_original"].head(15).tolist()
        kw_blob = ", ".join(kws)
        prompt = prompts[sys_lang].format(keywords=kw_blob)
        try:
            chat = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = chat.choices[0].message.content
            # Try to parse JSON
            data = {}
            try:
                data = json.loads(content)
            except Exception:
                # crude extraction of JSON-like
                import re as _re
                m = _re.search(r"\{.*\}", content, _re.S)
                if m:
                    data = json.loads(m.group(0))
            cluster_name = data.get("cluster_name") or f"Cluster {cl_id}"
            description = data.get("description") or ""
        except Exception:
            cluster_name, description = f"Cluster {cl_id}", ""
        results.append((cl_id, cluster_name, description))

    name_df = pd.DataFrame(results, columns=["cluster_id", "cluster_name", "cluster_description"])
    return df.merge(name_df, on="cluster_id", how="left")

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Keyword Clustering (ES/EN/PT)", layout="wide")

st.title("🔎 Keyword Clustering (ES · EN · PT)")
st.caption("Carga tu CSV, normalizamos los términos y agrupamos por similitud. Listo para GitHub/Streamlit Cloud.")

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
    use_openai = st.checkbox("Nombrar clústeres con OpenAI", value=False)
    if use_openai:
        openai_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    else:
        openai_key = ""

st.markdown("### 1) Sube tu CSV de keywords")
uploaded = st.file_uploader("CSV con columna de keywords", type=["csv"])

if uploaded is None:
    st.info("💡 Consejo: la columna debe llamarse `keyword`, `keywords`, `query` o `kw` (o será la primera columna).")
    st.stop()

# Read CSV robustly
try:
    df = pd.read_csv(uploaded, encoding="utf-8-sig", on_bad_lines="skip", dtype=str)
except Exception:
    uploaded.seek(0)
    df = pd.read_csv(uploaded, encoding="latin-1", on_bad_lines="skip", dtype=str)

df = df.fillna("")
possible = [c for c in df.columns if c.lower() in {"keyword", "keywords", "query", "kw"}]
kw_col = possible[0] if possible else df.columns[0]

# Keep original and normalized
df["keyword_original"] = df[kw_col].astype("string")
df["keyword_norm"] = normalize_series(df[kw_col])
df = df[df["keyword_norm"].str.len() > 0].drop_duplicates(subset=["keyword_norm"]).reset_index(drop=True)

st.success(f"✅ CSV cargado. Usando columna: **{kw_col}** · Filas: **{len(df)}**")
with st.expander("Ver muestra de normalización", expanded=False):
    st.dataframe(df[["keyword_original", "keyword_norm"]].head(20))

# Build vectors
texts = df["keyword_norm"].tolist()

X = None
vectorizer = None
nlp = None

if embed_method == "spaCy vectors":
    nlp = build_spacy_pipeline(lang_choice)
    if nlp is not None and nlp.vocab.vectors.shape[0] > 0:
        with st.spinner("Calculando vectores de spaCy..."):
            X = embed_spacy(texts, nlp)
            # Some spaCy small models yield mostly-zero vectors; check variance
            if np.allclose(X, 0):
                st.warning("Los vectores de este modelo parecen nulos. Cambio a TF-IDF.")
                X, vectorizer = embed_tfidf(texts)
    else:
        st.warning("No se pudo cargar un modelo de spaCy válido. Uso TF-IDF.")
        X, vectorizer = embed_tfidf(texts)
else:
    X, vectorizer = embed_tfidf(texts)

# Decide K
if auto_k:
    with st.spinner("Buscando K óptimo (silhouette)..."):
        k_auto = try_auto_k(X, k_min=2, k_max=min(12, max(3, len(df)//5)))
        if k_auto:
            k = k_auto
st.write(f"**K seleccionado:** {k}")

# Cluster
labels = kmeans_cluster(X, k)
df["cluster_id"] = labels

# Optional PCA visualization
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

# Optional OpenAI naming
if use_openai and openai_key:
    with st.spinner("Nombrando clústeres con OpenAI..."):
        df = name_clusters_with_openai(df, api_key=openai_key, lang="Auto")
else:
    df["cluster_name"] = ""
    df["cluster_description"] = ""

# Output
st.markdown("### 3) Resultado")
order_cols = ["keyword_original", "keyword_norm", "cluster_id", "cluster_name", "cluster_description"]
extra_cols = [c for c in df.columns if c not in order_cols]
df = df[order_cols + extra_cols]
st.dataframe(df)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV clusterizado", data=csv_bytes, file_name="clustered_keywords.csv", mime="text/csv")

st.caption("Hecho con ❤️ para es/en/pt · Normalización robusta para evitar fallos por caracteres/acentos.")
