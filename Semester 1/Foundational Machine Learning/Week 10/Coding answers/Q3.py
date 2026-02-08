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


def grad_mse(X, y, w):
    """
    Compute the gradient of J(w) = ||Xw - y||^2 w.r.t. w:
        ∇_w J(w) = 2 X^T (Xw - y)
    
    Parameters
    ----------
    X : np.ndarray, shape (N, M)
    y : np.ndarray, shape (N,)
    w : np.ndarray, shape (M,)
    
    Returns
    -------
    grad : np.ndarray, shape (M,)
        Gradient of the squared-error objective.
    """
    # Compute residuals (Xw - y)
    residuals = X @ w - y
    
    # Use the formula from the slides: grad = 2 X^T (Xw - y)
    grad = 2 * (X.T @ residuals)
    return grad


def fit_ml_gd(X, y, lr=1e-2, n_iters=1000):
    """
    Fit linear regression with gradient descent, approximating the ML solution.
    
    Parameters
    ----------
    X : np.ndarray, shape (N, M)
        Design matrix.
    y : np.ndarray, shape (N,)
        Targets.
    lr : float
        Learning rate.
    n_iters : int
        Number of gradient descent iterations.
    
    Returns
    -------
    w : np.ndarray, shape (M,)
        Final weight vector after gradient descent.
    losses : list of float
        MSE loss value at each iteration.
    """
    N, M = X.shape
    
    # Initialise weights at zero
    w = np.zeros(M)
    
    # List to store the loss at each iteration (for monitoring convergence)
    losses = []
    
    for t in range(n_iters):
        # Compute gradient of the squared-error objective
        grad = grad_mse(X, y, w)
        
        # Gradient descent update: w <- w - lr * grad
        w = w - lr * grad
        
        # Compute current MSE loss for monitoring
        loss = mse_loss(X, y, w)
        losses.append(loss)
        
        # Optionally print progress every 200 iterations
        if (t + 1) % 200 == 0:
            print(f"Iter {t+1:4d}, MSE: {loss:.4f}")
    
    return w, losses


# -------- Compare GD solution with closed-form ML --------
if __name__ == "__main__":
    # Generate synthetic data
    X, y, w_true = generate_linear_data(
        n_samples=200,
        n_features=5,
        noise_std=0.3,
        seed=1
    )
    
    # Closed-form ML solution
    w_ml = fit_ml(X, y)
    
    # Gradient descent approximation to ML
    w_gd, losses = fit_ml_gd(X, y, lr=1e-3, n_iters=3000)
    
    print("\nTrue weights:\n", w_true)
    print("Closed-form ML weights:\n", w_ml)
    print("GD-approx ML weights:\n", w_gd)
    
    # Compare errors
    print("\nL2 error (ML vs true):", np.linalg.norm(w_ml - w_true))
    print("L2 error (GD vs true):", np.linalg.norm(w_gd - w_true))
    print("L2 difference (GD vs ML):", np.linalg.norm(w_gd - w_ml))