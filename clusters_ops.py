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

def update_distance_matrix(distance_matrix, clusters, X, complete=True):
    """
    Tính toán và cập nhật lại ma trận khoảng cách giữa các cụm hiện hành.
    Vì tham số có truyền vào X (dữ liệu gốc), nên ta sẽ tính trực tiếp khoảng cách 
    từ các điểm trong cụm.
    """
    # Lấy danh sách các ID cụm đang còn tồn tại
    active_cluster_ids = list(clusters.keys())
    n_active = len(active_cluster_ids)
    
    # Khởi tạo ma trận khoảng cách mới với kích thước là số cụm đang còn lại
    new_dist_matrix = np.zeros((n_active, n_active))
    
    for i in range(n_active):
        for j in range(n_active):
            if i == j:
                # Cụm với chính nó thì cho khoảng cách vô cực để hàm find_closets_clusters bỏ qua
                new_dist_matrix[i, j] = np.inf
            else:
                # Lấy danh sách index các điểm dữ liệu của cụm i và cụm j
                points_i = clusters[active_cluster_ids[i]]
                points_j = clusters[active_cluster_ids[j]]
                
                # Tính Complete-Linkage: lấy khoảng cách LỚN NHẤT giữa mọi cặp điểm
                if complete == True:
                    max_distance = 0
                    for p1 in points_i:
                        for p2 in points_j:
                            # Tính khoảng cách Euclidean giữa 2 điểm
                            dist = np.linalg.norm(X[p1] - X[p2])
                            if dist > max_distance:
                                max_distance = dist
                                
                    new_dist_matrix[i, j] = max_distance
                    new_dist_matrix[j, i] = max_distance # Ma trận đối xứng
                    
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