import numpy as np

def generate_linear_data(n_samples, n_features, noise_std, seed=None):
    """
    Generate synthetic data from a linear regression model:
        y = X w_true + noise,   noise ~ N(0, noise_std^2 I)
    
    Parameters
    ----------
    n_samples : int
        Number of data points N.
    n_features : int
        Number of features/dimensions M.
    noise_std : float
        \sigma Standard deviation of the Gaussian noise.
    seed : int or None
        Random seed for reproducibility. If None, use global RNG state.
    
    Returns
    -------
    X : np.ndarray, shape (N, M)
        Design matrix of input features.
    y : np.ndarray, shape (N,)
        Target vector.
    w_true : np.ndarray, shape (M,)
        The true underlying weight vector used to generate y.
    """
    if seed is not None:
        # Fix random seed so that results are reproducible
        np.random.seed(seed)
    
    # Sample the true weight vector from a standard normal distribution.
    # Each entry in w_true is drawn i.i.d. from N(0, 1).
    w_true = np.random.randn(n_features)
    
    # Sample the design matrix X with standard normal entries.
    # Each row x_i represents the feature vector for one data point.
    X = np.random.randn(n_samples, n_features)
    
    # Generate Gaussian noise epsilon ~ N(0, noise_std^2 I).
    noise = noise_std * np.random.randn(n_samples)
    # Alternatively, np.random.normal can be used to generate noise.
    # noise = np.random.normal(loc=0, scale=noise_std, size=n_samples)
    
    # Generate targets: y = X w_true + noise.
    y = X @ w_true + noise
    
    return X, y, w_true


# -------- Small test of the function --------
if __name__ == "__main__":
    X, y, w_true = generate_linear_data(
        n_samples=5,
        n_features=3,
        noise_std=0.5,
        seed=0
    )
    print("X:\n", X)
    print("y:\n", y)
    print("w_true:\n", w_true)