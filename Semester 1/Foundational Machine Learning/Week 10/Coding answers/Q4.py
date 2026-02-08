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

def fit_map(X, y, sigma2):
    """
    Compute MAP solution for linear regression with prior w ~ N(0, I).
    
    Objective:
        J_MAP(w) = ||Xw - y||^2 + sigma2 * ||w||^2
    
    Closed-form solution:
        w_map = (X^T X + sigma2 I)^{-1} X^T y
    
    Parameters
    ----------
    X : np.ndarray, shape (N, M)
        Design matrix.
    y : np.ndarray, shape (N,)
        Targets.
    sigma2 : float
        Noise variance (sigma^2), also scaling the L2 penalty term.
    
    Returns
    -------
    w_map : np.ndarray, shape (M,)
        MAP estimate of the weights.
    """
    N, M = X.shape
    
    # Compute X^T X (M x M)
    XT_X = X.T @ X
    
    # Form the regularised matrix A = X^T X + sigma2 I_M
    A = XT_X + sigma2 * np.eye(M)
    
    # Compute X^T y
    XT_y = X.T @ y
    
    # Solve (X^T X + sigma2 I) w = X^T y for w
    w_map = np.linalg.solve(A, XT_y)
    
    return w_map


# -------- Compare ML and MAP in a "small N, large M" setting --------
if __name__ == "__main__":
    np.random.seed(10)
    
    # Small number of samples but many features
    N = 30   # samples
    M = 50   # features
    
    X, y, w_true = generate_linear_data(
        n_samples=N,
        n_features=M,
        noise_std=0.5,
        seed=10
    )
    
    # ML solution (may be unstable if X^T X is ill-conditioned)
    try:
        w_ml = fit_ml(X, y)
    except np.linalg.LinAlgError:
        # If X^T X is singular, fall back to pseudo-inverse
        w_ml = np.linalg.pinv(X.T @ X) @ X.T @ y
    
    # Try several sigma^2 values to see the effect of regularisation
    for sigma2 in [0.1, 1.0, 10.0]:
        w_map = fit_map(X, y, sigma2)
        print(f"\nSigma^2 = {sigma2}")
        print("||w_ml||    =", np.linalg.norm(w_ml))
        print("||w_map||   =", np.linalg.norm(w_map))
        print("||w_true||  =", np.linalg.norm(w_true))
        print("ML vs true  =", np.linalg.norm(w_ml - w_true))
        print("MAP vs true =", np.linalg.norm(w_map - w_true))