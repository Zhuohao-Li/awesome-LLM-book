import numpy as np

def k_means_clustering(data: np.ndarray, k: int) -> np.ndarray:
    np.random.seed(0)  # Setting the seed for reproducibility

    # Step 1: Initialize centroids randomly from the data points
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    # To store the value of centroids when it updates
    old_centroids = np.zeros(centroids.shape)

    # Cluster labels(0, 1, 2, ...)
    clusters = np.zeros(data.shape[0])

    # Error func. - Distance between new centroids and old centroids
    error = np.linalg.norm(centroids - old_centroids)

    # Loop will run till the error becomes zero
    while error != 0:
    ###### your code here: ##### 
        # ---- (A) 分配：将每个样本分配给最近的质心 ----
        # 距离矩阵： shape = (n_samples, k)
        # 利用广播：对每个样本与每个质心计算欧氏距离
        dists = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        clusters = np.argmin(dists, axis=1)

        # ---- (B) 更新：按簇求均值得到新的质心 ----
        old_centroids = centroids.copy()
        for j in range(k):
            mask = (clusters == j)
            if np.any(mask):
                centroids[j] = data[mask].mean(axis=0)
            else:
                # 空簇处理：若某簇没有样本，随机重置该质心为某个样本点
                centroids[j] = data[np.random.randint(0, data.shape[0])]

        # ---- (C) 计算收敛误差 ----
        error = np.linalg.norm(centroids - old_centroids)

    return clusters

    # return clusters.astype(int)