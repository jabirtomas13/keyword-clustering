#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword Clustering App (ES/EN/PT) – SEO & EEAT Edition
------------------------------------------------------
Esta versión incluye las mejoras solicitadas para:
1. Embeddings Semánticos Profundos (SBERT) para entender la intención real.
2. Prompt de Gemini especializado en EEAT y Content Pillars.
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

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="SEO Content Clusterizer (EEAT)", layout="wide")

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
# Embeddings: SpaCy & SBERT (El motor semántico)
# ------------------------
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
            st.info(f"⬇️ Descargando modelo spaCy '{name}'...")
            subprocess.run([sys.executable, "-m", "spacy", "download", name], check=True)
            return spacy.load(name)
        except Exception: return None

    nlp = _load(target)
    if nlp: return nlp
    if fallback and fallback != target:
        st.warning(f"⚠️ Falló '{target}'. Intentando '{fallback}'...")
        nlp = _load(fallback)
        if nlp: return nlp
    st.error("❌ Fallo crítico en spaCy. Se usará TF-IDF.")
    return None

def embed_spacy(texts: List[str], nlp) -> np.ndarray:
    vecs = []
    bar = st.progress(0)
    tot = len(texts)
    for i, t in enumerate(texts):
        if i % 50 == 0: bar.progress((i+1)/tot)
        vecs.append(nlp(t).vector)
    bar.empty()
    return np.vstack(vecs)

def embed_sbert(texts: List[str]) -> Optional[np.ndarray]:
    """Genera embeddings contextuales profundos usando Sentence Transformers."""
    STClass = _lazy_import_sbert()
    if not STClass:
        st.error("❌ Librería 'sentence-transformers' no instalada.")
        st.info("💡 Ejecuta: `pip install sentence-transformers`")
        return None

    # Modelo ligero pero muy potente para semántica multilingüe
    model_name = 'all-MiniLM-L6-v2'
    try:
        with st.spinner(f"⏳ Cargando modelo SBERT ({model_name})... la primera vez tarda un poco."):
            model = STClass(model_name)
            embeddings = model.encode(texts, show_progress_bar=True)
            return embeddings
    except Exception as e:
        st.error(f"Error cargando SBERT: {e}")
        return None

def embed_tfidf(texts: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
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
            # Conversión segura para silhouette
            X_data = X.toarray() if hasattr(X, "toarray") else X
            score = silhouette_score(X_data, labels, metric="cosine")
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue
    return best_k or 5

# ------------------------
# Nombrado con Gemini (SEO Expert Mode - EEAT)
# ------------------------
def name_clusters_seo_mode(df: pd.DataFrame, api_key: str, lang: str) -> pd.DataFrame:
    genai = _lazy_import_gemini()
    if not genai:
        st.warning("Google GenAI no disponible.")
        return df

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Error Gemini Auth: {e}")
        return df

    # Configuración de idioma y prompt
    sys_lang = {"Auto": "en", "English": "en", "Español": "es", "Português": "pt"}.get(lang.split(" ")[0], "en")

    # PROMPT AVANZADO PARA EEAT / GEN AI
    seo_prompts = {
        "es": """Actúa como un Experto SEO Senior especializado en EEAT y Generative AI.
        Analiza el siguiente grupo de keywords.
        Tu objetivo es definir la estrategia para un 'Pilar de Contenido' de alta autoridad.

        Keywords: {keywords}

        Devuelve SOLO un JSON con:
        1. "cluster_name": Título H1 optimizado para el artículo pilar (máx 6 palabras).
        2. "user_intent": El objetivo real del usuario (ej: 'Informacional - Resolver problema técnico', 'Transaccional - Comprar software').
        3. "content_angle": En una frase, ¿cuál es el enfoque único para ganar autoridad en este tema?
        """,
        "en": """Act as a Senior SEO Expert specialized in EEAT and GenAI.
        Analyze this keyword cluster.
        Your goal is to define a strategy for a high-authority 'Content Pillar'.

        Keywords: {keywords}

        Return ONLY JSON with:
        1. "cluster_name": Optimized H1 Title for the pillar page (max 6 words).
        2. "user_intent": The specific user intent (e.g., 'Informational - Troubleshooting', 'Transactional - Buying Guide').
        3. "content_angle": One sentence on the unique angle to establish authority.
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
    total = len(groups)

    bar = st.progress(0)
    status = st.empty()

    for i, (cid, group) in enumerate(groups):
        if i > 0: time.sleep(2) # Rate limit suave
        bar.progress((i+1)/total)
        status.text(f"Analizando Cluster {cid} con enfoque SEO (EEAT)...")

        kws = group["keyword_original"].head(20).tolist()
        prompt = prompt_tmpl.format(keywords=", ".join(kws))

        c_name, c_intent, c_angle = f"Cluster {cid}", "N/A", "N/A"

        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt],
                    config=gen_config
                )
                data = json.loads(resp.text)
                c_name = data.get("cluster_name", c_name)
                c_intent = data.get("user_intent", "")
                c_angle = data.get("content_angle", "")
                break
            except Exception as e:
                if "429" in str(e) and attempt == 0:
                    time.sleep(5)
                    continue
                else:
                    break

        results.append((cid, c_name, c_intent, c_angle))

    bar.empty()
    status.empty()

    res_df = pd.DataFrame(results, columns=["cluster_id", "cluster_name", "user_intent", "content_angle"])
    return df.merge(res_df, on="cluster_id", how="left")

# ------------------------
# UI Principal
# ------------------------
st.title("🧠 SEO Content Clusterizer (EEAT & GenAI Ready)")
st.markdown("""
Esta herramienta agrupa keywords basándose en **intención semántica profunda** (SBERT) y utiliza AI para definir pilares de contenido de alta autoridad.
""")

with st.sidebar:
    st.header("⚙️ Configuración")

    embed_method = st.radio(
        "Tecnología de Embeddings",
        ["SBERT (Alta Calidad - Recomendado)", "spaCy vectors (Rápido)", "TF-IDF (Básico)"],
        index=0,
        help="SBERT entiende el contexto completo de la frase, ideal para distinguir intenciones."
    )

    lang_choice = "auto-multi (xx)"
    if embed_method == "spaCy vectors (Rápido)":
        lang_choice = st.selectbox("Idioma spaCy", ["auto-multi (xx)", "english (en)", "español (es)", "português (pt)"])

    auto_k = st.checkbox("K Automático (Silhouette)", value=True)
    k = st.slider("Clusters Manuales (K)", 2, 50, 10)

    st.divider()
    use_gemini = st.checkbox("Análisis SEO con Gemini", value=False)
    gemini_key = ""
    if use_gemini:
        gemini_key = st.text_input("GEMINI_API_KEY", type="password", value=os.getenv("GEMINI_API_KEY", ""))

st.markdown("### 1) Sube Data")
uploaded = st.file_uploader("CSV Keywords", type=["csv"])

if not uploaded:
    st.info("Sube un CSV con una columna llamada `keyword`.")
    st.stop()

# Carga CSV
try:
    df = pd.read_csv(uploaded, encoding="utf-8-sig", on_bad_lines="skip", dtype=str)
except:
    uploaded.seek(0)
    df = pd.read_csv(uploaded, encoding="latin-1", on_bad_lines="skip", dtype=str)

df = df.fillna("")
pos = [c for c in df.columns if c.lower() in {"keyword", "keywords", "query", "kw"}]
kw_col = pos[0] if pos else df.columns[0]
df["keyword_original"] = df[kw_col].astype("string")
df["keyword_norm"] = normalize_series(df[kw_col])
df = df[df["keyword_norm"].str.len() > 0].drop_duplicates(subset=["keyword_norm"]).reset_index(drop=True)

st.success(f"Procesando **{len(df)}** keywords.")

# ------------------------
# Generación de Embeddings
# ------------------------
texts = df["keyword_norm"].tolist()
X = None

if "SBERT" in embed_method:
    X = embed_sbert(texts)
    if X is None:
        st.warning("⚠️ Falló SBERT. Cayendo a TF-IDF.")
        X, _ = embed_tfidf(texts)

elif "spaCy" in embed_method:
    nlp = build_spacy_pipeline(lang_choice)
    if nlp:
        with st.spinner("Vectorizando con spaCy..."):
            X = embed_spacy(texts, nlp)
            if np.allclose(X, 0): X, _ = embed_tfidf(texts)
    else:
        X, _ = embed_tfidf(texts)
else:
    X, _ = embed_tfidf(texts)

# ------------------------
# Clustering & Análisis
# ------------------------
if auto_k:
    with st.spinner("Calculando coherencia de tópicos (K óptimo)..."):
        # Ajuste de K máximo según cantidad de datos
        limit = min(20, max(3, len(df)//3))
        k = try_auto_k(X, k_min=3, k_max=limit)
st.metric("Clusters Identificados", k)

df["cluster_id"] = kmeans_cluster(X, k)

# Visualización
st.markdown("### 2) Mapa de Intenciones (2D)")
try:
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_dense)
    viz = pd.DataFrame({
        "x": coords[:,0], "y": coords[:,1],
        "cluster": df["cluster_id"],
        "keyword": df["keyword_original"]
    })
    st.scatter_chart(viz, x="x", y="y", color="cluster")
except Exception as e:
    st.write("Datos insuficientes para visualizar.")

# Gemini SEO Analysis
if use_gemini and gemini_key:
    # LLAMADA CLAVE PARA EEAT
    df = name_clusters_seo_mode(df, gemini_key, "Español") 
else:
    df["cluster_name"] = "Cluster " + df["cluster_id"].astype(str)
    df["user_intent"] = ""
    df["content_angle"] = ""

# Output
st.markdown("### 3) Estrategia de Contenidos (EEAT)")
cols = ["cluster_id", "cluster_name", "user_intent", "content_angle", "keyword_original"]
final_df = df[cols].sort_values(["cluster_id"])

st.dataframe(
    final_df,
    column_config={
        "cluster_name": st.column_config.TextColumn("Pilar de Contenido (H1)"),
        "user_intent": st.column_config.TextColumn("Intención"),
        "content_angle": st.column_config.TextColumn("Enfoque de Autoridad"),
        "keyword_original": st.column_config.ListColumn("Keywords")
    },
    use_container_width=True
)

st.download_button(
    "⬇️ Descargar Estrategia CSV",
    final_df.to_csv(index=False).encode("utf-8"),
    "seo_strategy_clusters.csv",
    "text/csv"
)
