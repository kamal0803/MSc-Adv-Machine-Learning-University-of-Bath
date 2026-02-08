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
    
def mse_loss(X, y, w):
    """
    Compute the mean squared error (MSE) loss:
        MSE = (1/N) * ||Xw - y||^2
    
    Parameters
    ----------
    X : np.ndarray, shape (N, M)
        Design matrix.
    y : np.ndarray, shape (N,)
        Targets.
    w : np.ndarray, shape (M,)
        Current weight vector.
    
    Returns
    -------
    mse : float
        Mean squared error.
    """
    # Predicted outputs y_pred = Xw
    y_pred = X @ w
    
    # Residuals are the differences between prediction and target
    residuals = y_pred - y
    
    # Sum of squared residuals ||Xw - y||^2
    squared_error = residuals @ residuals  # same as np.sum(residuals**2)
    
    # Divide by N to get the mean squared error
    N = X.shape[0]
    mse = squared_error / N
    return mse
    
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
    
def train_val_split(X, y, val_ratio=0.2, seed=None):
    """
    Split (X, y) into train and validation sets.
    
    Parameters
    ----------
    X : np.ndarray, shape (N, M)
    y : np.ndarray, shape (N,)
    val_ratio : float
        Fraction of data to use for validation.
    seed : int or None
        Random seed for shuffling.
    
    Returns
    -------
    X_train, y_train, X_val, y_val
    """
    if seed is not None:
        # Fix the random seed so that the split is reproducible
        np.random.seed(seed)
    
    N = X.shape[0]
    indices = np.arange(N)
    
    # Shuffle indices to randomise which points go to train vs validation
    np.random.shuffle(indices)
    
    # Compute number of validation samples
    n_val = int(N * val_ratio)
    
    # First n_val indices are used for validation
    val_idx = indices[:n_val]
    # The rest are used for training
    train_idx = indices[n_val:]
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    
    return X_train, y_train, X_val, y_val


# -------- Hyperparameter tuning via validation --------
if __name__ == "__main__":
    # 1. Generate a dataset from the linear model
    X, y, w_true = generate_linear_data(
        n_samples=300,
        n_features=10,
        noise_std=0.4,
        seed=123
    )
    
    # 2. Split into training and validation sets
    X_train, y_train, X_val, y_val = train_val_split(
        X, y,
        val_ratio=0.3,
        seed=123
    )
    
    # 3. Try a range of sigma^2 values
    sigma2_list = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    print("Tuning sigma^2 using validation MSE:\n")
    
    best_sigma2 = None
    best_val_mse = np.inf
    
    for sigma2 in sigma2_list:
        if sigma2 == 0.0:
            # sigma^2 = 0 corresponds to no regularisation (pure ML)
            w = fit_ml(X_train, y_train)
        else:
            # Use MAP with given sigma^2
            w = fit_map(X_train, y_train, sigma2)
        
        # Compute training and validation MSEs
        train_mse = mse_loss(X_train, y_train, w)
        val_mse = mse_loss(X_val, y_val, w)
        
        print(f"sigma^2 = {sigma2:6.2f} | "
              f"train MSE = {train_mse:8.4f} | "
              f"val MSE = {val_mse:8.4f}")
        
        # Keep track of the best sigma^2 according to validation MSE
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_sigma2 = sigma2
    
    print("\nBest sigma^2 according to validation MSE:", best_sigma2)
    print("Best validation MSE:", best_val_mse)