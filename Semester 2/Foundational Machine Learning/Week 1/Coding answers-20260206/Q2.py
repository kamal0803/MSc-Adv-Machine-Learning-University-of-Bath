import numpy as np

def kmeans_best_of_n(X, K, n_init=10, max_iter=100, tol=1e-6, base_seed=0):
    """
    Run k-means multiple times and choose the result with the lowest WCSS.

    Parameters
    ----------
    X : array-like, shape (N, M)
    K : int
    n_init : int
        Number of independent initialisations.
    max_iter, tol : forwarded to kmeans_from_scratch
    base_seed : int
        Base seed; each run uses base_seed + i.

    Returns
    -------
    best_centroids : np.ndarray, shape (K, M)
    best_labels : np.ndarray, shape (N,)
    best_wcss : float
    """
    X = np.asarray(X, dtype=float)

    best_centroids = None
    best_labels = None
    best_wcss = np.inf

    # Run k-means with different seeds, inducing different initial centroid choices.
    for i in range(n_init):
        seed = base_seed + i

        # Call the k-means implementation from Q1.
        centroids, labels, wcss = kmeans_from_scratch(
            X, K, max_iter=max_iter, tol=tol, seed=seed
        )

        # Keep the solution with the smallest objective value.
        if wcss < best_wcss:
            best_wcss = wcss
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels, float(best_wcss)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Create two clusters for demonstration
    X1 = rng.normal(loc=(0, 0), scale=0.5, size=(50, 2))
    X2 = rng.normal(loc=(5, 5), scale=0.5, size=(50, 2))
    X = np.vstack([X1, X2])

    centroids, labels, wcss = kmeans_best_of_n(X, K=2, n_init=20, base_seed=100)
    print("Best WCSS:", wcss)