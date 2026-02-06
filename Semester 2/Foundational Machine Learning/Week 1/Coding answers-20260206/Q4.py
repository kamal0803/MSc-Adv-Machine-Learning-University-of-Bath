import numpy as np

def euclidean(a, b):
    """Euclidean distance between two vectors a and b."""
    return float(np.sqrt(np.sum((a - b) ** 2)))

def single_link_distance(X, cluster_a, cluster_b):
    """
    Single-link distance between two clusters:
    the minimum distance between any point in cluster_a and any point in cluster_b.
    """
    best = float("inf")
    for i in cluster_a:
        for j in cluster_b:
            d = euclidean(X[i], X[j])
            if d < best:
                best = d
    return best

def agglomerative_single_linkage(X):
    """
    Easy-to-follow agglomerative clustering (single linkage).

    Returns
    -------
    merges : list of (a_id, b_id, dist, new_id)
        a_id and b_id are the cluster IDs merged at distance dist,
        producing a new cluster with ID new_id.
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]

    # Start with N singleton clusters: {0}, {1}, ..., {N-1}
    clusters = {i: {i} for i in range(N)}  # cluster_id -> set of point indices
    next_id = N
    merges = []

    # Keep merging until only one cluster remains
    while len(clusters) > 1:
        ids = list(clusters.keys())

        # Find the closest pair of clusters
        best_dist = float("inf")
        best_pair = None

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_id = ids[i]
                b_id = ids[j]
                d = single_link_distance(X, clusters[a_id], clusters[b_id])
                if d < best_dist:
                    best_dist = d
                    best_pair = (a_id, b_id)

        # Merge the closest pair
        a_id, b_id = best_pair
        new_cluster = clusters[a_id] | clusters[b_id]

        new_id = next_id
        next_id += 1

        # Record this merge event
        merges.append((a_id, b_id, best_dist, new_id))

        # Update active clusters
        del clusters[a_id]
        del clusters[b_id]
        clusters[new_id] = new_cluster

    return merges


if __name__ == "__main__":
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [2.0, 2.0],
        [2.1, 2.0],
    ])

    merges = agglomerative_single_linkage(X)
    for a_id, b_id, dist, new_id in merges:
        print(f"Merged {a_id} and {b_id} at dist={dist:.3f} -> new cluster {new_id}")