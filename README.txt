
project/
|
├── main.py              # chạy toàn bộ pipeline (không bao gồm bước tiền xử lí)
├── clustering.py        # xử lý thuật toán hierarchical clustering 
|
├── utils.py             # hàm phụ trợ cho AgglomerativeClusteringCustom
|
|
└── README.md            # mô tả project


Thiết kế class với tên:
  AgglomerativeClusteringCustom
Các phương thức cần có:
  __init__(self, n_clusters)
  fit(self, X)
  compute_distance_matrix(self, X) (nó là proximity matrix nhưng đổi tên:v) - nhớ là hàm này dùng Complete-Linkage
  find_closest_clusters(self)
  update_distance_matrix(self, cluster1, cluster2)
  get_labels(self)
