import numpy as np

def kmeans_from_scratch(X, K, max_iter=100, tol=1e-6, seed=0):
    """
    Simple, easy-to-follow k-means (NumPy only).

    Returns
    -------
    centroids : (K, M) array
    labels    : (N,) array of ints in {0,...,K-1}
    wcss      : float, sum of squared distances to assigned centroid
    """
    # ---------- Step 0: basic setup ----------
    X = np.asarray(X, dtype=float)
    N, M = X.shape
    rng = np.random.default_rng(seed)

    # ---------- Step 1: initialise centroids ----------
    # Pick K random data points as initial centroids
    centroids = X[rng.choice(N, size=K, replace=False)].copy()

    # ---------- Step 2: repeat assignment + update ----------
    for _ in range(max_iter):
        # (A) Assignment: compute distance to each centroid, pick nearest
        # dists_sq[i, k] = ||X[i] - centroids[k]||^2
        dists_sq = np.zeros((N, K))
        for k in range(K):
            diff = X - centroids[k]              # (N, M)
            dists_sq[:, k] = np.sum(diff**2, axis=1)

        labels = np.argmin(dists_sq, axis=1)     # (N,)

        # (B) Update: recompute each centroid as mean of its assigned points
        new_centroids = centroids.copy()
        for k in range(K):
            points_k = X[labels == k]
            if len(points_k) > 0:
                new_centroids[k] = points_k.mean(axis=0)
            else:
                # If a cluster becomes empty, re-pick a random data point
                new_centroids[k] = X[rng.integers(0, N)]

        # (C) Convergence: stop if centroids barely move
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    # ---------- Step 3: compute WCSS ----------
    # Sum of squared distances of each point to its assigned centroid
    wcss = 0.0
    for i in range(N):
        diff = X[i] - centroids[labels[i]]
        wcss += float(np.sum(diff**2))

    return centroids, labels, wcss


if __name__ == "__main__":
    X = np.array([[0.0, 0.0],
                  [0.2, 0.1],
                  [3.0, 3.0],
                  [3.1, 2.9],
                  [10.0, 10.0]])
    centroids, labels, wcss = kmeans_from_scratch(X, K=2, seed=42)
    print("Centroids:\n", centroids)
    print("Labels:", labels)
    print("WCSS:", wcss)