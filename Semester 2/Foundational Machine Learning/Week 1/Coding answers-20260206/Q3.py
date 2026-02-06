import numpy as np

def clustering_quality_metrics(X, labels, centroids):
    """
    Simple, easy-to-follow clustering metrics:

    1) avg_intra: average distance from each point to its assigned centroid
    2) min_inter: minimum distance between any two centroids
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    centroids = np.asarray(centroids, dtype=float)

    N = X.shape[0]
    K = centroids.shape[0]

    # -----------------------------
    # 1) Average intra-cluster distance
    # -----------------------------
    # For each point i, compute distance to its assigned centroid.
    total = 0.0
    for i in range(N):
        c = centroids[labels[i]]          # centroid of point i
        diff = X[i] - c
        dist = np.sqrt(np.sum(diff**2))   # Euclidean distance
        total += float(dist)
    avg_intra = total / N

    # -----------------------------
    # 2) Minimum inter-centroid distance
    # -----------------------------
    # Compute distance between every pair of centroids and take the smallest.
    min_inter = float("inf")
    for i in range(K):
        for j in range(i + 1, K):
            diff = centroids[i] - centroids[j]
            dist = np.sqrt(np.sum(diff**2))
            if dist < min_inter:
                min_inter = float(dist)

    return avg_intra, min_inter


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))

    # Example: two fixed centroids and nearest-centroid labels
    centroids = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        d0 = np.sum((X[i] - centroids[0])**2)
        d1 = np.sum((X[i] - centroids[1])**2)
        labels[i] = 0 if d0 <= d1 else 1

    avg_intra, min_inter = clustering_quality_metrics(X, labels, centroids)
    print("Average intra distance:", avg_intra)
    print("Minimum inter-centroid distance:", min_inter)