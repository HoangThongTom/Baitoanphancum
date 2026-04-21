import numpy as np

def find_closets_clusters(distance_matrix):
    """
    Tìm 2 cụm có khoảng cách nhỏ nhất trong ma trận khoảng cách.
    (Giữ nguyên tên hàm 'closets' theo yêu cầu đề bài)
    """
    min_dist = np.inf
    c1 = -1
    c2 = -1
    
    # Lấy kích thước ma trận
    n_rows, n_cols = distance_matrix.shape
    
    # Duyệt qua từng ô trong ma trận để tìm giá trị nhỏ nhất
    for i in range(n_rows):
        for j in range(n_cols):
            # Bỏ qua khoảng cách của 1 cụm tới chính nó (thường là 0 hoặc inf)
            if i != j:
                if distance_matrix[i, j] < min_dist:
                    min_dist = distance_matrix[i, j]
                    c1 = i
                    c2 = j
                    
    return c1, c2

def merge_clusters(clusters, i, j):
    """
    Gộp cụm j vào cụm i. 
    Giả sử 'clusters' là một dictionary lưu danh sách các điểm dữ liệu, ví dụ: {0: [0], 1: [1, 2]}
    """
    # Lấy toàn bộ các điểm dữ liệu từ cụm j bỏ sang cụm i
    clusters[i].extend(clusters[j])
    
    # Xóa cụm j khỏi dictionary vì nó đã bị gộp
    del clusters[j]
    
    return clusters
def update_distance_matrix(clusters, X, complete=True):
    n = len(clusters)
    new_dist_matrix = np.full((n, n), np.inf)

    for i in range(n):
        for j in range(i + 1, n):

            points_i = clusters[i]
            points_j = clusters[j]

            if complete:
                dist = max(
                    np.linalg.norm(X[p1] - X[p2])
                    for p1 in points_i
                    for p2 in points_j
                )

            new_dist_matrix[i, j] = dist
            new_dist_matrix[j, i] = dist

    return new_dist_matrix
def record_merge(history, c1, c2, dist):
    """
    Lưu lại lịch sử gộp cụm để theo dõi hoặc vẽ Dendrogram sau này.
    """
    # Tạo một tuple hoặc list chứa thông tin 2 cụm vừa gộp và khoảng cách giữa chúng
    merge_info = (c1, c2, dist)
    
    # Thêm vào danh sách lịch sử
    history.append(merge_info)
    
    return history