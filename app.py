import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import spacy
import subprocess
import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from openai import OpenAI

# Helper function to load or download spaCy models
def load_or_download_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", model_name])
        return spacy.load(model_name)

# Sidebar: API Key Input
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

if not api_key:
    st.error("Please provide your OpenAI API key in the sidebar.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Sidebar: Language selection
lang = st.sidebar.selectbox("Select a language for processing:", ("en", "es", "pt"))

# Load spaCy model based on selected language
model_map = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "pt": "pt_core_news_sm"
}
nlp = load_or_download_model(model_map[lang])

# Language-specific prompt templates
prompt_templates = {
    "en": """Given the following keywords, identify a common theme and return:
1. A short cluster name (max 5 words)
2. A one-sentence description.
Keywords: {keywords}
Respond in **JSON format** like this:
{{
  "cluster_name": "Descriptive Name",
  "description": "Brief explanation of the category."
}}""",
    "es": """Dadas las siguientes palabras clave, identifica un tema común y devuelve:
1. Un nombre corto para el grupo (máx. 5 palabras)
2. Una breve descripción en una oración.
Palabras clave: {keywords}
Responde en formato **JSON** así:
{{
  "cluster_name": "Nombre descriptivo",
  "description": "Breve explicación de la categoría."
}}""",
    "pt": """Dadas as palavras-chave a seguir, identifique um tema comum e retorne:
1. Um nome curto para o grupo (máx. 5 palavras)
2. Uma breve descrição em uma frase.
Palavras-chave: {keywords}
Responda no formato **JSON** assim:
{{
  "cluster_name": "Nome descritivo",
  "description": "Breve explicação da categoria."
}}"""
}

# Main UI
st.title("Keyword Clustering Tool")

uploaded_file = st.file_uploader("Upload a CSV file with keywords", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None, names=["keyword"])
    df = df.dropna(subset=["keyword"])
    df["keyword"] = df["keyword"].astype(str).str.strip()

    def preprocess_keywords(keywords):
        processed_keywords = []
        for keyword in keywords:
            doc = nlp(keyword.lower().strip())
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            processed = ' '.join(tokens).strip()
            processed_keywords.append(processed if processed else keyword)
        return processed_keywords

    st.info("Processing keywords...")
    keywords = preprocess_keywords(df["keyword"].tolist())
    df['keyword_processed'] = keywords
    df = df[df['keyword_processed'] != ""]

    # Generate embeddings
    st.info("Generating embeddings with OpenAI...")

    BATCH_SIZE = 100
    SLEEP_SECONDS = 1

    def batch_embed(texts, model="text-embedding-ada-002"):
        embeddings = []
        for start in range(0, len(texts), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch = texts[start:end]
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )
                embeddings += [item.embedding for item in response.data]
                time.sleep(SLEEP_SECONDS)
            except Exception as e:
                st.error(f"Error embedding batch {start}-{end}: {e}")
                embeddings += [[0] * 1536] * len(batch)
        return np.array(embeddings)

    try:
        keyword_embeddings = batch_embed(df['keyword_processed'].tolist())
        st.success(f"Generated embeddings for {len(keyword_embeddings)} keywords.")
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        st.stop()

    # Dimensionality Reduction
    pca = PCA(n_components=min(100, len(keyword_embeddings), keyword_embeddings.shape[1]))
    keyword_embeddings = pca.fit_transform(keyword_embeddings)

    # Clustering
    NUM_CLUSTERS = min(25, len(df))
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(keyword_embeddings)
    df["cluster_id"] = clusters

    # Generate Cluster Names
    def generate_cluster_name_and_description(keywords):
        prompt = prompt_templates[lang].format(keywords=', '.join(keywords[:25]))
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            response_text = response.choices[0].message.content.strip()
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(response_text)
            return data.get("cluster_name", "Uncategorized"), data.get("description", "No description provided.")
        except Exception as e:
            st.error(f"Error generating cluster name: {e}")
            return "Uncategorized", "Error occurred."

    df['cluster_name'] = ''
    df['cluster_description'] = ''

    for cluster_num in range(NUM_CLUSTERS):
        cluster_keywords = df[df['cluster_id'] == cluster_num]['keyword_processed'].tolist()
        if len(cluster_keywords) < 5:
            cluster_name, cluster_description = "Uncategorized", "Too few keywords to determine"
        else:
            cluster_name, cluster_description = generate_cluster_name_and_description(cluster_keywords)
        df.loc[df['cluster_id'] == cluster_num, 'cluster_name'] = cluster_name
        df.loc[df['cluster_id'] == cluster_num, 'cluster_description'] = cluster_description

    # Display and download
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Clustered Keywords as CSV",
        data=csv,
        file_name='auto_clustered_keywords.csv',
        mime='text/csv',
    )
