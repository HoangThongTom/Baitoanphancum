
project/
|
├── main.py              # chạy toàn bộ pipeline (không bao gồm bước tiền xử lí)
├── clustering.py        # xử lý thuật toán hierarchical clustering 
|
├── utils.py             # hàm phụ trợ cho AgglomerativeClusteringCustom
|
|
└── README.md            # mô tả project



---

# 🧠 `__init__(n_clusters)`

* Lưu số cụm cần dừng
* Khởi tạo:

  * danh sách clusters
  * distance matrix
  * labels

---

# 🚀 `fit(X)` (quan trọng nhất)

* Khởi tạo mỗi điểm = 1 cluster
* Tính distance matrix ban đầu
* Lặp đến khi còn `n_clusters`:

  * tìm 2 cluster gần nhất
  * gộp 2 cluster
  * cập nhật clusters
  * cập nhật distance matrix (complete-linkage)
* Gán nhãn cuối:

  * mỗi cluster → 1 label

---

# 📏 `compute_distance_matrix(X)`

* Tính ma trận khoảng cách NxN giữa các điểm
* Thường dùng Euclidean
* Ban đầu chỉ là point-to-point (chưa phải cluster)

---

# 🔍 `find_closest_clusters()`

* Duyệt ma trận khoảng cách hiện tại
* Tìm cặp cluster có distance nhỏ nhất
* Không chọn i == j
* Trả về index 2 cluster cần merge

---

# 🔄 `update_distance_matrix(c1, c2)` (core của complete-linkage)

* Xoá 2 cluster khỏi matrix
* Tạo cluster mới = union(c1, c2)
* Với mỗi cluster còn lại:

  * distance mới = MAX(distance(c1, C), distance(c2, C))
* Cập nhật lại matrix đồng bộ với clusters

---

# 🏷️ `get_labels()`

* Trả về mảng nhãn cuối cùng
* Mỗi cluster gán 1 id
* Mỗi điểm thuộc cluster nào thì nhận label đó

---

# ⚠️ Phần bắt buộc phải quản lý thêm

* Danh sách clusters luôn đồng bộ với matrix
* Index thay đổi sau mỗi lần merge
* Fit phải là “bộ điều khiển chính”
* Complete-link luôn dùng **MAX distance**

---

# ✔️ Tóm lại

đây là lời của chatGPT, mọi người có thể làm theo cách khác
