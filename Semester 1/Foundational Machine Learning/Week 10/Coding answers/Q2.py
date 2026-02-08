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

def fit_ml(X, y):
    """
    Compute the Maximum Likelihood (ML) solution for linear regression:
        w_ml = (X^T X)^{-1} X^T y
    
    Parameters
    ----------
    X : np.ndarray, shape (N, M)
        Design matrix.
    y : np.ndarray, shape (N,)
        Target vector.
    
    Returns
    -------
    w_ml : np.ndarray, shape (M,)
        ML estimate of the weight vector.
    """
    # Step 1: Compute X^T X, the MxM feature covariance matrix.
    XT_X = X.T @ X
    
    # Step 2: Compute X^T y, the M-dimensional vector of correlations
    # between each feature and the target.
    XT_y = X.T @ y
    
    # Step 3: Solve the linear system (X^T X) w = X^T y.
    # This implements the normal equations from the slides.
    # Using np.linalg.solve is typically more stable than explicitly
    # computing the matrix inverse.
    w_ml = np.linalg.solve(XT_X, XT_y)
    
    return w_ml


# -------- Demonstration using Exercise 1's generator --------
if __name__ == "__main__":
    # Generate synthetic data from the known model
    X, y, w_true = generate_linear_data(
        n_samples=200,
        n_features=5,
        noise_std=0.3,
        seed=42
    )
    
    # Compute the closed-form ML solution
    w_ml = fit_ml(X, y)
    
    print("True weights:\n", w_true)
    print("ML estimate:\n", w_ml)
    
    # Compute the L2 (Euclidean) error between w_ml and w_true.
    error = np.linalg.norm(w_ml - w_true)
    print("L2 error ||w_ml - w_true||:", error)