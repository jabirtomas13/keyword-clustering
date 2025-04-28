import os
 import time
 import json
 import numpy as np
 import pandas as pd
 from getpass import getpass
 from openai import OpenAI
 import nltk
 import spacy
 from nltk.corpus import stopwords
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.decomposition import PCA
 from scipy.cluster.hierarchy import linkage, fcluster
 from sklearn.metrics.pairwise import cosine_similarity
 
 # Install dependencies (uncomment if needed)
 # !pip install openai pandas numpy scikit-learn nltk spacy tqdm sentence-transformers
 # !python -m spacy download en_core_web_sm
 
 # Download NLTK resources
 nltk.download('stopwords', quiet=True)
 nltk.download('wordnet', quiet=True)
 
 # Load SpaCy model
 try:
     nlp = spacy.load("en_core_web_sm")
 except OSError:
     print("Downloading SpaCy model...")
     os.system("python -m spacy download en_core_web_sm")
     nlp = spacy.load("en_core_web_sm")
 
 # ✅ Step 1: Set up API Key securely
 os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")
 client = OpenAI()
 
 # ✅ Step 2: Load and Preprocess Dataset
 CSV_FILE = "keywords.csv"
 try:
     df = pd.read_csv(CSV_FILE, header=None, names=["keyword"])
     keywords = df["keyword"].tolist()
     print(f"✅ Loaded {len(keywords)} keywords from {CSV_FILE}")
 except FileNotFoundError:
     print(f"❌ File not found: {CSV_FILE}")
     exit(1)
 
 # Preprocess keywords
 stop_words = set(stopwords.words('english'))
 
 def preprocess_keywords(keywords):
     processed_keywords = []
     for keyword in keywords:
         if not isinstance(keyword, str):
             processed_keywords.append("")
             continue
 
         doc = nlp(keyword.lower())  # Process keyword with SpaCy
         tokens = [token.text for token in doc if token.is_alpha and token.text.lower() not in stop_words]
 
         # Preserve named entities
         entities = [ent.text for ent in doc.ents]
 
         processed_text = ' '.join(tokens) if tokens else keyword.lower()
         processed_keywords.append(processed_text)
     return processed_keywords
 
 print("Preprocessing keywords...")
 keywords_processed = preprocess_keywords(keywords)
 df['keyword_processed'] = keywords_processed
 print("✅ Keywords preprocessed.")
 
 # SEMANTIC IMPROVEMENT 1: Use Sentence-BERT locally for better embeddings
 # This provides high-quality semantic embeddings without API costs
 try:
     from sentence_transformers import SentenceTransformer
     print("Using Sentence-BERT for high-quality local embeddings...")
 
     # Load a lightweight model that's good for keyword-style short text
     model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
 
     # Generate embeddings locally
     embeddings = model.encode(df['keyword_processed'].fillna(''), show_progress_bar=True)
     keyword_embeddings = embeddings
     print(f"✅ Generated {len(keyword_embeddings)} embeddings using Sentence-BERT")
 
     USE_SBERT = True
 except ImportError:
     print("Sentence-BERT not available, falling back to OpenAI embeddings")
     USE_SBERT = False
 
 # If Sentence-BERT is not available, use OpenAI embeddings selectively
 if not USE_SBERT:
     # ✅ Generate embeddings using a cost-effective approach
     # SEMANTIC IMPROVEMENT 2: Use a representative sample strategy for large datasets
     def sample_and_embed_keywords(keywords, sample_size=1000, model="text-embedding-ada-002"):
         """Intelligently sample keywords and generate embeddings"""
         N = len(keywords)
 
         if N <= sample_size:
             # For small datasets, embed everything
             sample_indices = list(range(N))
             full_sample = keywords
         else:
             # For large datasets, use intelligent sampling:
             # 1. Split keywords into chunks and take representatives from each
             chunk_size = N // (sample_size // 2)
             sample_indices = []
 
             # Get keywords from each chunk
             for i in range(0, N, chunk_size):
                 chunk = keywords[i:i+chunk_size]
                 # Take ~2 keywords from each chunk
                 sample_count = max(1, min(2, len(chunk) // 5))
                 chunk_indices = list(range(i, min(i+chunk_size, N)))
                 if chunk_indices:
                     selected = np.random.choice(chunk_indices, sample_count, replace=False)
                     sample_indices.extend(selected)
 
             # 2. Add some completely random keywords to ensure diversity
             remaining = sample_size - len(sample_indices)
             if remaining > 0:
                 remaining_indices = list(set(range(N)) - set(sample_indices))
                 if remaining_indices:
                     random_indices = np.random.choice(remaining_indices,
                                                     min(remaining, len(remaining_indices)),
                                                     replace=False)
                     sample_indices.extend(random_indices)
 
             full_sample = [keywords[i] for i in sample_indices]
 
         # Generate embeddings for the sample
         try:
             response = client.embeddings.create(
                 model=model,
                 input=[k for k in full_sample if k]
             )
             sample_embeddings = np.array([item.embedding for item in response.data])
             print(f"✅ Generated embeddings for {len(sample_embeddings)} sample keywords")
 
             return sample_embeddings, sample_indices
         except Exception as e:
             print(f"❌ Error generating embeddings: {e}")
             return None, sample_indices
 
     # Generate embeddings for a sample
     sample_size = min(1000, len(df))
     sample_embeddings, sample_indices = sample_and_embed_keywords(
         df['keyword_processed'].tolist(),
         sample_size=sample_size
     )
 
     if sample_embeddings is None:
         print("Falling back to TF-IDF vectorization...")
         vectorizer = TfidfVectorizer(max_features=300)
         tfidf_matrix = vectorizer.fit_transform(df['keyword_processed'].fillna(''))
         keyword_embeddings = tfidf_matrix.toarray()
     else:
         # SEMANTIC IMPROVEMENT 3: Propagate embeddings to similar keywords
         print("Propagating embeddings to similar keywords...")
         # Create TF-IDF matrix for all keywords
         vectorizer = TfidfVectorizer()
         tfidf_matrix = vectorizer.fit_transform(df['keyword_processed'].fillna(''))
 
         # For each non-sampled keyword, find the most similar sampled keyword and use its embedding
         keyword_embeddings = np.zeros((len(df), sample_embeddings.shape[1]))
         for i in sample_indices:
             keyword_embeddings[i] = sample_embeddings[sample_indices.index(i)]
 
         # For non-sampled keywords, find the most similar sampled keyword
         for i in range(len(df)):
             if i not in sample_indices:
                 similarities = cosine_similarity(
                     tfidf_matrix[i:i+1],
                     tfidf_matrix[sample_indices]
                 )[0]
                 most_similar_idx = sample_indices[np.argmax(similarities)]
                 keyword_embeddings[i] = keyword_embeddings[most_similar_idx]
 
         print(f"✅ Propagated embeddings to all {len(keyword_embeddings)} keywords")
 
 # ✅ Step 4b: Apply Dimensionality Reduction (PCA)
 try:
     max_components = min(50, keyword_embeddings.shape[0], keyword_embeddings.shape[1])
     pca = PCA(n_components=max_components)
     keyword_embeddings = pca.fit_transform(keyword_embeddings)
     print(f"✅ Dimensionality reduction applied (PCA) to {max_components} dimensions.")
 except Exception as e:
     print(f"❌ Error with PCA: {e}")
     exit(1)
 
 # ✅ Step 5: Clustering using Hierarchical Clustering
 num_clusters = min(20, len(df))  # Ensure we don't create more clusters than data points
 try:
     Z = linkage(keyword_embeddings, method="ward")
     df["cluster_id"] = fcluster(Z, t=num_clusters, criterion="maxclust")
     print(f"✅ Keywords successfully clustered into {num_clusters} groups.")
 except Exception as e:
     print(f"❌ Error in clustering: {e}")
     exit(1)
 
 # SEMANTIC IMPROVEMENT 4: Generate representative keywords for each cluster
 def get_representative_keywords(df, cluster_id, n=10):
     """Get the most representative keywords for a cluster based on embedding similarity"""
     cluster_df = df[df['cluster_id'] == cluster_id]
     if len(cluster_df) <= n:
         return cluster_df['keyword'].tolist()
 
     # Calculate centroid of the cluster
     indices = cluster_df.index.tolist()
     cluster_embeddings = np.array([keyword_embeddings[i] for i in indices])
     centroid = np.mean(cluster_embeddings, axis=0)
 
     # Calculate distance to centroid for each keyword
     distances = [np.linalg.norm(emb - centroid) for emb in cluster_embeddings]
 
     # Get indices of keywords closest to centroid
     closest_indices = np.argsort(distances)[:n]
     representative_keywords = [cluster_df.iloc[i]['keyword'] for i in closest_indices]
 
     return representative_keywords
 
 # SEMANTIC IMPROVEMENT 5: Generate better cluster names and descriptions
 # with a two-step process that first understands the cluster, then names it
 def generate_improved_cluster_names(clusters_with_representatives):
     """
     Two-step process for better cluster naming:
     1. First analyze each cluster's representative keywords
     2. Then generate a name and description based on that analysis
     """
     if not clusters_with_representatives:
         return {}
 
     results = {}
 
     # Process clusters in smaller batches to manage costs
     batch_size = 5
     for batch_start in range(0, len(clusters_with_representatives), batch_size):
         batch_end = min(batch_start + batch_size, len(clusters_with_representatives))
         batch_clusters = list(clusters_with_representatives.items())[batch_start:batch_end]
 
         try:
             # Step 1: Analyze clusters
             analysis_prompt = "I'll provide representative keywords for several clusters. For each cluster, analyze the keywords to identify common themes, topics, or categories.\n\n"
 
             for cluster_id, keywords in batch_clusters:
                 analysis_prompt += f"Cluster {cluster_id} representative keywords: {', '.join(keywords[:15])}\n\n"
 
             analysis_response = client.chat.completions.create(
                 model="gpt-3.5-turbo",  # Use cheaper model for analysis
                 messages=[{"role": "user", "content": analysis_prompt}],
                 temperature=0.1,
                 max_tokens=300
             )
 
             analysis_text = analysis_response.choices[0].message.content.strip()
 
             # Step 2: Generate names and descriptions based on analysis
             naming_prompt = f"""Based on the following analysis of keyword clusters, provide a specific name and description for each cluster.
 
 Analysis:
 {analysis_text}
 
 For each cluster, provide:
 1. A short, specific cluster name (3-5 words)
 2. A one-sentence description that accurately represents the theme
 
 Respond in JSON format:
 {{
   "clusters": [
     {{
       "cluster_id": 1,
       "cluster_name": "...",
       "description": "..."
     }},
     ...
   ]
 }}
 """
 
             naming_response = client.chat.completions.create(
                 model="gpt-3.5-turbo",
                 messages=[{"role": "user", "content": naming_prompt}],
                 temperature=0.2,
                 max_tokens=200,
                 response_format={"type": "json_object"}
             )
 
             naming_text = naming_response.choices[0].message.content.strip()
             data = json.loads(naming_text)
 
             # Extract results
             for cluster_info in data.get("clusters", []):
                 cluster_id_str = str(cluster_info.get("cluster_id"))
                 # Match string cluster ID to our numeric cluster IDs
                 for actual_id, _ in batch_clusters:
                     if str(actual_id) == cluster_id_str:
                         results[actual_id] = (
                             cluster_info.get("cluster_name", f"Cluster {actual_id}"),
                             cluster_info.get("description", "No description provided.")
                         )
 
             # Fill in any missing clusters from this batch
             for cluster_id, _ in batch_clusters:
                 if cluster_id not in results:
                     results[cluster_id] = (f"Cluster {cluster_id}", "No description generated")
 
             print(f"Processed cluster name batch {batch_start//batch_size + 1}")
             time.sleep(1)
 
         except Exception as e:
             print(f"Error generating names for batch starting at {batch_start}: {e}")
             # Provide default names for all clusters in this batch
             for cluster_id, _ in batch_clusters:
                 results[cluster_id] = (f"Cluster {cluster_id}", "Error generating description")
 
     return results
 
 # Get representative keywords for each cluster
 print("Identifying representative keywords for each cluster...")
 clusters_with_representatives = {}
 for cluster_num in df['cluster_id'].unique():
     representatives = get_representative_keywords(df, cluster_num, n=15)
     clusters_with_representatives[cluster_num] = representatives
     print(f"Cluster {cluster_num}: Found {len(representatives)} representative keywords")
 
 # Generate improved cluster names
 print("Generating semantically improved cluster names and descriptions...")
 cluster_names = generate_improved_cluster_names(clusters_with_representatives)
 
 # Apply results to dataframe
 df['cluster_name'] = ''
 df['cluster_description'] = ''
 df['representative'] = False
 
 for cluster_num, (name, description) in cluster_names.items():
     df.loc[df['cluster_id'] == cluster_num, 'cluster_name'] = name
     df.loc[df['cluster_id'] == cluster_num, 'cluster_description'] = description
 
     # Mark representative keywords
     for keyword in clusters_with_representatives.get(cluster_num, []):
         matching_indices = df[(df['cluster_id'] == cluster_num) & (df['keyword'] == keyword)].index
         if not matching_indices.empty:
             df.loc[matching_indices, 'representative'] = True
 
     print(f"Cluster {cluster_num}: {name} - {description}")
 
 # SEMANTIC IMPROVEMENT 6: Calculate semantic coherence scores for each cluster
 def calculate_cluster_coherence(cluster_embeddings):
     """Calculate semantic coherence of a cluster based on embedding similarity"""
     if len(cluster_embeddings) <= 1:
         return 1.0  # Perfect coherence for single element
 
     # Calculate centroid
     centroid = np.mean(cluster_embeddings, axis=0)
 
     # Calculate average cosine similarity to centroid
     similarities = []
     for emb in cluster_embeddings:
         similarity = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid))
         similarities.append(similarity)
 
     return np.mean(similarities)
 
 # Calculate coherence for each cluster
 print("Calculating semantic coherence for each cluster...")
 df['cluster_coherence'] = 0.0
 
 for cluster_num in df['cluster_id'].unique():
     cluster_indices = df[df['cluster_id'] == cluster_num].index.tolist()
     cluster_embeddings = np.array([keyword_embeddings[i] for i in cluster_indices])
     coherence = calculate_cluster_coherence(cluster_embeddings)
     df.loc[df['cluster_id'] == cluster_num, 'cluster_coherence'] = coherence
     print(f"Cluster {cluster_num}: Coherence = {coherence:.3f}")
 
 # ✅ Step 7: Save the Results
 try:
     output_file = "semantic_clustered_keywords.csv"
     df.to_csv(output_file, index=False)
     print(f"✅ Semantic clustering complete! Results saved to '{output_file}'.")
 
     # Print summary
     print("\nClustering Summary:")
     for cluster_num in df['cluster_id'].unique():
         count = len(df[df['cluster_id'] == cluster_num])
         name = df[df['cluster_id'] == cluster_num]['cluster_name'].iloc[0]
         coherence = df[df['cluster_id'] == cluster_num]['cluster_coherence'].iloc[0]
         print(f"Cluster {cluster_num}: {name} - {count} keywords (Coherence: {coherence:.3f})")
 
         # Print some representative examples
         print("  Representative keywords:")
         representatives = df[(df['cluster_id'] == cluster_num) & (df['representative'] == True)]['keyword'].tolist()[:5]
         for rep in representatives:
             print(f"    - {rep}")
 except Exception as e:
     print(f"❌ Error saving results: {e}")
