# 1. Import thư viện
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


from metrics import manual_silhouette_score, manual_davies_bouldin_score, manual_calinski_harabasz_score
from model import AgglomerativeClusteringCustom

# 2. Load dữ liệu
data = load_iris()
X = data.data
y_true = data.target

df = pd.DataFrame(X, columns=data.feature_names)

print("Shape:", df.shape)
print(df.head())

# 3. Kiểm tra dữ liệu
print("\nMissing values:\n", df.isnull().sum())

# 4. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Khỏi tạo mô hình 
for n_clusters in range(2, 11):
    model = AgglomerativeClusteringCustom(
        n_clusters=n_clusters
    )

    labels = model.fit_predict(X_scaled)

    sil = manual_silhouette_score(X_scaled, labels)
    db  = manual_davies_bouldin_score(X_scaled, labels)
    ch  = manual_calinski_harabasz_score(X_scaled, labels)
    
    print(f"k={n_clusters} | Silhouette={sil:.4f} | DB={db:.4f} | CH={ch:.4f}")


plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.title("Cluster visualization")
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')

for c in np.unique(labels):
    points = X[labels == c]
    centroid = points.mean(axis=0)
    plt.scatter(centroid[0], centroid[1], s=200, marker='X')

plt.title("Clusters + centroids")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='rainbow')

plt.show()

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='rainbow')
plt.title("PCA visualization of clusters")
plt.show()

def plot_dendrogram(X, method="complete"):
    """
    Vẽ dendrogram cho hierarchical clustering
    """
    Z = linkage(X, method=method)

    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title(f"Dendrogram ({method} linkage)")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()

plot_dendrogram(X)