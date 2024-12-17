from src.utils import *
import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, conver_tol  =0.001,  max_iter=300):
        self.n_clusters = n_clusters # số cụm
        self.conver_tol = conver_tol # Ngưỡng hội tụ 
        self.max_iter = max_iter # Số lần lặp tối đa
        self.cluster_centers_ = None # Tâm cụm
        self.labels_ = None # Nhãn của các điểm dữ liệu
        self.inertia_ = None # Tổng bình phương khoảng cách từ các điểm dữ liệu đến tâm cụm gần nhất
        self.n_iter_ = None # Số lần lặp để hội tụ
    
    def _init_centroids(self, data):
        # Chọn ngẫu nhiên các điểm dữ liệu làm tâm cụm
        return data[np.random.choice(data.shape[0], self.n_clusters, replace=False)]

    def assign_labels(self, data, centroids):
        # Tính khoảng cách từ mỗi điểm dữ liệu đến các tâm cụm
        dist = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        # Gán nhãn cho mỗi điểm dữ liệu bằng cụm gần nhất
        return np.argmin(dist, axis=1)
    
    def update_centroids(self, data, labels):
        # Cập nhật tâm cụm bằng trung bình của các điểm dữ liệu trong cùng một cụm
        return np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])
    
    def has_converged(self, new_centroids, old_centroids):
        # Kiểm tra điều kiện hội tụ
        return np.linalg.norm(new_centroids - old_centroids) <= self.conver_tol
    
    def _calculate_inertia(self, data, centroids, labels):
        # Tính tổng bình phương khoảng cách từ các điểm dữ liệu đến tâm cụm gần nhất
        return np.sum(np.linalg.norm(data - centroids[labels], axis=1) ** 2)
    
    def predict(self, data):
        # Gán nhãn cho các điểm dữ liệu
        return self.assign_labels(data, self.cluster_centers_)
    
    def transform(self, data):
        # Tính khoảng cách từ các điểm dữ liệu đến các tâm cụm
        return np.linalg.norm(data[:, np.newaxis] - self.cluster_centers_, axis=2)
    
    def fit_transform(self, data):
        # Huấn luyện mô hình và gán nhãn cho các điểm dữ liệu
        self.fit(data)
        return self.transform(data)
    
    def fit_predict(self, data):
        # Huấn luyện mô hình và gán nhãn cho các điểm dữ liệu
        self.fit(data)
        return self.predict(data)
    
    def fit(self, data):
        # Lưu dữ liệu
        self.data = data
        # Khởi tạo tâm cụm
        self.cluster_centers_ = self._init_centroids(data)
        # Khởi tạo nhãn
        self.labels_ = np.zeros(data.shape[0])
        # Khởi tạo tổng bình phương khoảng cách
        self.inertia_ = 0
        # Khởi tạo số lần lặp
        self.n_iter_ = 0
        # Lặp cho đến khi hội tụ
        while True:
            # Gán nhãn cho các điểm dữ liệu
            self.labels_ = self.assign_labels(data, self.cluster_centers_)
            # Cập nhật tâm cụm
            new_cluster_centers = self.update_centroids(data, self.labels_)
            # Tính tổng bình phương khoảng cách
            new_inertia = self._calculate_inertia(data, new_cluster_centers, self.labels_)
            # Tăng số lần lặp
            self.n_iter_ += 1
            # Kiểm tra điều kiện hội tụ
            if self.has_converged(new_cluster_centers, self.cluster_centers_) or self.n_iter_ >= self.max_iter:
                self.cluster_centers_ = new_cluster_centers
                self.inertia_ = new_inertia
                break
            self.cluster_centers_ = new_cluster_centers
            self.inertia_ = new_inertia
        return
    
    def __str__(self):
        # Get distances to cluster centers
        display(Markdown("**Distances to cluster centers:**\n"))
        print(self.transform(self.data), "\n")

        # Print cluster centers
        display(Markdown("**Cluster Centers:**\n"))
        print(self.cluster_centers_, "\n")

        # Print inertia
        display(Markdown("**Inertia:**\n"))
        print(self.inertia_, "\n")

        # Print cluster labels
        display(Markdown("**Cluster labels:**\n"))
        print(self.labels_, "\n")
        return ""
    


