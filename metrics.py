import numpy as np
from itertools import combinations

# Adjusted Rand Index (ARI)

def adjusted_rand_index(labels_true, labels_pred):
    """
    Tính Adjusted Rand Index (ARI) giữa nhãn thực và nhãn dự đoán.
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    classes_true  = np.unique(labels_true)
    classes_pred  = np.unique(labels_pred)

    # Xây dựng contingency matrix
    contingency = np.zeros((len(classes_true), len(classes_pred)), dtype=np.int64)
    for i, c_true in enumerate(classes_true):
        for j, c_pred in enumerate(classes_pred):
            contingency[i, j] = np.sum((labels_true == c_true) & (labels_pred == c_pred))

    # Tổng theo hàng và cột
    row_sums = contingency.sum(axis=1)  # a_i
    col_sums = contingency.sum(axis=0)  # b_j
    n = labels_true.shape[0]

    # C(n, 2) helper
    def comb2(x):
        return x * (x - 1) / 2

    # Tính các thành phần
    sum_comb_c  = np.sum(comb2(contingency))   # Σ C(n_ij, 2)
    sum_comb_a  = np.sum(comb2(row_sums))       # Σ C(a_i,  2)
    sum_comb_b  = np.sum(comb2(col_sums))       # Σ C(b_j,  2)
    comb_n      = comb2(n)                      # C(n, 2)

    # Expected index
    expected = (sum_comb_a * sum_comb_b) / comb_n if comb_n > 0 else 0

    # Max index
    max_index = (sum_comb_a + sum_comb_b) / 2

    denominator = max_index - expected
    if denominator == 0:
        return 1.0 if sum_comb_c == expected else 0.0

    ari = (sum_comb_c - expected) / denominator
    return float(ari)


# Normalized Mutual Information (NMI)

def normalized_mutual_information(labels_true, labels_pred):
    """
    Tính Normalized Mutual Information (NMI) giữa nhãn thực và nhãn dự đoán.
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    n = len(labels_true)

    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)

    # Xây dựng contingency matrix
    contingency = np.zeros((len(classes_true), len(classes_pred)), dtype=np.float64)
    for i, c_true in enumerate(classes_true):
        for j, c_pred in enumerate(classes_pred):
            contingency[i, j] = np.sum((labels_true == c_true) & (labels_pred == c_pred))

    # Xác suất
    p_true = contingency.sum(axis=1) / n   # P(U = i)
    p_pred = contingency.sum(axis=0) / n   # P(V = j)
    p_joint = contingency / n              # P(U=i, V=j)

    # Entropy H(U) và H(V)
    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    H_true = entropy(p_true)
    H_pred = entropy(p_pred)

    # Mutual Information MI(U, V)
    mi = 0.0
    for i in range(len(classes_true)):
        for j in range(len(classes_pred)):
            if p_joint[i, j] > 0:
                mi += p_joint[i, j] * np.log(p_joint[i, j] / (p_true[i] * p_pred[j]))

    denominator = H_true + H_pred
    if denominator == 0:
        return 1.0

    nmi = 2 * mi / denominator
    return float(nmi)


# Silhouette Score

def silhouette_score(X, labels):
    """
    Tính Silhouette Score trung bình cho toàn bộ tập dữ liệu.
    """
    X = np.array(X, dtype=np.float64)
    labels = np.array(labels)
    n = X.shape[0]
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError("Silhouette Score cần ít nhất 2 cluster.")

    # Tính ma trận khoảng cách Euclidean NxN
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    scores = np.zeros(n)

    for i in range(n):
        label_i = labels[i]

        # a(i): khoảng cách trung bình đến các điểm cùng cluster
        same_cluster = np.where(labels == label_i)[0]
        same_cluster = same_cluster[same_cluster != i]

        if len(same_cluster) == 0:
            # Chỉ có 1 điểm trong cluster → s(i) = 0
            scores[i] = 0.0
            continue

        a_i = np.mean(dist_matrix[i, same_cluster])

        # b(i): khoảng cách trung bình nhỏ nhất đến cluster khác
        b_i = np.inf
        for other_label in unique_labels:
            if other_label == label_i:
                continue
            other_cluster = np.where(labels == other_label)[0]
            mean_dist = np.mean(dist_matrix[i, other_cluster])
            if mean_dist < b_i:
                b_i = mean_dist

        # s(i) - Đã sửa lỗi chia cho 0
        max_val = max(a_i, b_i)
        scores[i] = (b_i - a_i) / max_val if max_val > 0 else 0.0

    return float(np.mean(scores))


# Hàm tổng hợp — gọi 1 lần in ra cả 3 metric

def evaluate_clustering(X, labels_true, labels_pred):
    """
    Tính và in ra cả 3 metric đánh giá kết quả phân cụm.
    """
    ari = adjusted_rand_index(labels_true, labels_pred)
    nmi = normalized_mutual_information(labels_true, labels_pred)
    sil = silhouette_score(X, labels_pred)

    print("=" * 40)
    print("        Đánh giá kết quả phân cụm")
    print("=" * 40)
    print(f"  Adjusted Rand Index (ARI) : {ari:.4f}")
    print(f"  Normalized Mutual Info    : {nmi:.4f}")
    print(f"  Silhouette Score          : {sil:.4f}")
    print("=" * 40)

    return {"ARI": ari, "NMI": nmi, "Silhouette": sil}