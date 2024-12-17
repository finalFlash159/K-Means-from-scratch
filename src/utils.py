import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import Markdown, display
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns



# def preprocess_data(data):


def scatter_with_n_clusters(data, n_clusters, labels, cluster_centers):
    """
    Display scatter plot with n_clusters colors
    """
    plt.figure(figsize=(10, 10))
    for i in range(n_clusters):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], label=f'Cluster {i + 1}')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='black', marker='x', label='Centroids')
    plt.legend()
    plt.show()

def pca_scatter_plot_2d(data, kmeans, n_components=2):
    """
    Display scatter plot of data after PCA in 2D
    """
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    centers_pca = pca.transform(kmeans.cluster_centers_)

    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(kmeans.labels_)
    palette = sns.color_palette("viridis", n_colors=len(np.unique(kmeans.labels_)))

    for label in unique_labels:
        plt.scatter(data_pca[kmeans.labels_ == label, 0], data_pca[kmeans.labels_ == label, 1], color=palette[label])

    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, c='red', marker='X', label='Cluster centers')
    plt.title('Scatter Plot of Clusters after PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

def pca_scatter_plot_3d(data, kmeans, n_components=3):
    """
    Display scatter plot of data after PCA in 3D
    """
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    centers_pca = pca.transform(kmeans.cluster_centers_)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(kmeans.labels_)
    palette = sns.color_palette("viridis", n_colors=len(np.unique(kmeans.labels_)))

    for label in unique_labels:
        ax.scatter(data_pca[kmeans.labels_ == label, 0], data_pca[kmeans.labels_ == label, 1], data_pca[kmeans.labels_ == label, 2], color=palette[label])

    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], centers_pca[:, 2], s=200, c='red', marker='X', label='Cluster centers')
    plt.title('Scatter Plot of Clusters after PCA')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.legend()
    plt.show()


"""
-------------- For Image Compression ------------------
"""

def display_original_image(image_path):
    """
    Display original image
    """
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.show()

# read image and reshape
def read_image(image_path):
    img = plt.imread(image_path)
    width = img.shape[0]
    height = img.shape[1]
    img = img.reshape(width*height, 3)
    img = np.array(img, dtype=float) # convert img to np.float
    return img, width, height

# display and save image
def display_and_save_img(img, width, height, clusters, labels):
    """
    Hiển thị và lưu ảnh với số màu bằng số cụm (clusters).
    
    Parameters:
    -----------
    img : np.array
        Mảng ảnh đầu vào (dạng phẳng, kích thước [n_pixels, 3]).
    width : int
        Chiều rộng của ảnh gốc.
    height : int
        Chiều cao của ảnh gốc.
    clusters : np.array
        Tâm cụm màu (dạng [n_clusters, 3]).
    labels : np.array
        Mảng nhãn gán cho từng pixel (kích thước [n_pixels]).
    n_clusters : int
        Số cụm màu (số màu nén).
    
    Returns:
    --------
    None
    """
    # Tạo ảnh mới bằng cách gán mỗi pixel với màu của tâm cụm tương ứng
    img2 = np.zeros_like(img)
    for i in range(len(img2)):
        img2[i] = clusters[labels[i]]
    
    # Chuyển đổi về kích thước gốc và định dạng uint8
    img2 = img2.reshape(width, height, 3)
    img2 = img2.astype(np.uint8)
    
    # Lưu ảnh đã nén với số cụm
    n_clusters = np.unique(labels).shape[0]
    filename = f'images/compressed_image_{n_clusters}clusters.jpg'
    plt.imsave(filename, img2)
    
    # Hiển thị ảnh đã nén
    plt.imshow(img2)
    plt.axis('off')  # Tắt trục khi hiển thị ảnh
    plt.show()

    print(f"Compressed image saved as: {filename}")
