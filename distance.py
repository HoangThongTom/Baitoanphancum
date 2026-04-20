import numpy as np

# hàm tính kc Euclide
def euc_dist(a,b):
    return np.sqrt(np.sum((a-b)**2))

# hàm tính liên kết
def compute_cluster_distance(X, ci, cj):
    max_dist = float('-inf') # gán gtri ban đầu là âm vô cực
    for k in ci:
        for l in cj:
            d = euc_dist(X[k], X[l])
            if d > max_dist:
                max_dist = d      
    return max_dist

# hàm tính ma trận khoảng cách
def compute_distance_matrix(X, clusters):
    # khởi tạo ma trận khoảng cách
    dist_matrix = np.zeros((len(clusters), len(clusters)))
    # tính kc giữa từng cặp cụm với nhau
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            dist = compute_cluster_distance(X, clusters[i], clusters[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix