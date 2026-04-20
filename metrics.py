import numpy as np

def manual_silhouette_score(X, labels):
    n_samples = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2: return 0
    
    s_scores = []
    for i in range(n_samples):
        # Lấy các điểm cùng cụm với i (loại bỏ chính i)
        same_cluster = X[labels == labels[i]]
        if len(same_cluster) > 1:
            # a_i: trung bình khoảng cách tới các điểm cùng cụm
            a_i = np.mean([np.linalg.norm(X[i] - p) for p in same_cluster if not np.array_equal(p, X[i])])
        else:
            a_i = 0
            
        # b_i: trung bình khoảng cách tới cụm lân cận "gần nhất"
        b_i_list = []
        for label in unique_labels:
            if label == labels[i]: continue
            other_cluster = X[labels == label]
            avg_dist_to_other = np.mean([np.linalg.norm(X[i] - p) for p in other_cluster])
            b_i_list.append(avg_dist_to_other)
        
        b_i = min(b_i_list)
        s_scores.append((b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) != 0 else 0)
            
    return np.mean(s_scores)

def manual_davies_bouldin_score(X, labels):
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2: return 0

    centroids = [np.mean(X[labels == label], axis=0) for label in unique_labels]
    # S_i: Độ phân tán (khoảng cách TB từ các điểm trong cụm tới tâm)
    S = []
    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        dist_to_centroid = np.mean([np.linalg.norm(p - centroids[i]) for p in cluster_points])
        S.append(dist_to_centroid)
        
    max_R = []
    for i in range(n_clusters):
        R_ij_list = [((S[i] + S[j]) / np.linalg.norm(centroids[i] - centroids[j])) 
                     for j in range(n_clusters) if i != j]
        max_R.append(max(R_ij_list))
        
    return np.mean(max_R)

def manual_calinski_harabasz_score(X, labels):
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2: return 0
    
    overall_mean = np.mean(X, axis=0)
    bcss = 0 # Between-cluster sum of squares
    wcss = 0 # Within-cluster sum of squares
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_mean = np.mean(cluster_points, axis=0)
        # BCSS
        bcss += len(cluster_points) * np.sum((cluster_mean - overall_mean)**2)
        # WCSS
        wcss += np.sum((cluster_points - cluster_mean)**2)
        
    if wcss == 0: return 0
    return (bcss / (n_clusters - 1)) / (wcss / (n_samples - n_clusters))