import pandas as pd
import numpy as np
import json
import nltk
import sys
import subprocess
import time
import streamlit as st
import spacy
import en_core_web_sm
import es_core_news_sm
import pt_core_news_sm
from getpass import getpass
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Check if packages are installed, install if needed
required_packages = ['openai', 'pandas', 'numpy', 'scikit-learn', 'nltk', 'spacy']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Descargar recursos de NLTK si fuera necesario (ya no usados para lematización)
nltk.download('stopwords')
nltk.download('wordnet')

# Selector de idioma en Streamlit
lang = st.selectbox("Selecciona un idioma para procesar:", ("en", "es", "pt"))

# Cargar modelo spaCy según idioma seleccionado
if lang == "en":
    nlp = en_core_web_sm.load()
elif lang == "es":
    nlp = es_core_news_sm.load()
elif lang == "pt":
    nlp = pt_core_news_sm.load()

# ✅ Step 1: Set up API Key securely
from openai import OpenAI
client = OpenAI(api_key=getpass("Enter your OpenAI API key: "))

# ✅ Step 2: Load and Preprocess Dataset
CSV_FILE = "keywords.csv"  # Cambiá el nombre si tu archivo es otro
df = pd.read_csv(CSV_FILE, header=None, names=["keyword"])

# Eliminar valores nulos y limpiar strings
df = df.dropna(subset=["keyword"])
df["keyword"] = df["keyword"].astype(str).str.strip()
keywords = df["keyword"].tolist()

# Preprocesamiento con spaCy
def preprocess_keywords(keywords):
    processed_keywords = []
    for keyword in keywords:
        doc = nlp(keyword.lower().strip())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        processed = ' '.join(tokens).strip()
        processed_keywords.append(processed if processed else keyword)
    return processed_keywords

keywords = preprocess_keywords(keywords)
df['keyword_processed'] = keywords
df = df[df['keyword_processed'] != ""]

# ✅ Step 3: Generate Embeddings using text-embedding-ada-002
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
            print(f"Error in batch {start}-{end}: {e}")
            embeddings += [[0] * 1536] * len(batch)

    return np.array(embeddings)

print("Generating embeddings for keywords...")
try:
    keyword_embeddings = batch_embed(df['keyword_processed'].tolist())
    print(f"✅ Embeddings generated for {len(keyword_embeddings)} keywords.")
except Exception as e:
    print(f"❌ OpenAI API Error: {e}")
    exit()

# ✅ Step 3b: Dimensionality Reduction (PCA)
pca = PCA(n_components=min(100, len(keyword_embeddings), keyword_embeddings.shape[1]))
keyword_embeddings = pca.fit_transform(keyword_embeddings)
print("✅ Dimensionality reduction applied (PCA).")

# ✅ Step 4: Clustering using K-Means
NUM_CLUSTERS = min(25, len(df))
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
clusters = kmeans.fit_predict(keyword_embeddings)
df["cluster_id"] = clusters

# ✅ Step 5: Generate Cluster Names & Descriptions
def generate_cluster_name_and_description(keywords):
    prompt = f"""Given the following keywords, identify a common theme and return:
1. A short cluster name (max 5 words)
2. A one-sentence description.
Keywords: {', '.join(keywords[:25])}
Respond in **JSON format** like this:
{{
  "cluster_name": "Descriptive Name",
  "description": "Brief explanation of the category."
}}
"""
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
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parsing Error: {e}. AI Response: {response_text}")
        return "Uncategorized", "Could not determine category."
    except Exception as e:
        print(f"❌ Error: {e}")
        return "Uncategorized", "Error occurred when generating cluster name."

print("Generating cluster names and descriptions...")
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
    print(f"Cluster {cluster_num}: {cluster_name} - {cluster_description}")
    time.sleep(1)

# ✅ Step 6: Save to CSV
df.to_csv("auto_clustered_keywords.csv", index=False)
print("✅ Clustering complete! Results saved to 'auto_clustered_keywords.csv'.")
