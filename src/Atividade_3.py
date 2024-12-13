# %%
from sklearn.datasets import make_blobs
from random import randint
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

df = None
# %%
def load_df():
  X,_ = make_blobs(n_samples=500, n_features=randint(5,10), centers=randint(3,12), cluster_std=1.5, center_box=(-50.0, 50.0))
  return pd.DataFrame(X)
# %%
if df is None:
  df = load_df()

df
# %% Analisando o conjunto de dados
df.head()

df.shape
# %% Normalizando os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
# %% Treinando modelos KMeans para diferentes números de clusters

silhouette_scores = []
davies_bouldin_scores = []
calinski_scores = []
n_clusters_range = range(2, 11)

for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_scaled)
    
    # Calculando as métricas
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(df_scaled, kmeans.labels_))
    calinski_scores.append(calinski_harabasz_score(df_scaled, kmeans.labels_))

# Armazenando os resultados
cluster_metrics = pd.DataFrame({
    'Clusters': n_clusters_range,
    'Silhouette Score': silhouette_scores,
    'Davies Bouldin Score': davies_bouldin_scores,
    'Calinski Harabasz Score': calinski_scores
})

cluster_metrics
# %% Visualizando as métricas
plt.figure(figsize=(10, 6))

# Silhouette Score
plt.subplot(1, 3, 1)
plt.plot(n_clusters_range, silhouette_scores, marker='o', color='blue')
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")

# Davies Bouldin Score
plt.subplot(1, 3, 2)
plt.plot(n_clusters_range, davies_bouldin_scores, marker='o', color='green')
plt.title("Davies Bouldin Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Davies Bouldin Score")

# Calinski Harabasz Score
plt.subplot(1, 3, 3)
plt.plot(n_clusters_range, calinski_scores, marker='o', color='red')
plt.title("Calinski Harabasz Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Calinski Harabasz Score")

plt.tight_layout()
plt.show()
# %% Aplicando o modelo final com o número de clusters escolhido
best_n_clusters = 4  # Exemplo, ajuste com base na análise anterior

kmeans_final = KMeans(n_clusters=best_n_clusters, random_state=42)
kmeans_final.fit(df_scaled)

# Adicionando os rótulos ao DataFrame original
df['Cluster'] = kmeans_final.labels_

# Visualizando os clusters em um gráfico (usando as 2 primeiras componentes principais)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df['Cluster'], cmap='viridis', s=50)
plt.title("Clusters KMeans (PCA-reduced data)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()