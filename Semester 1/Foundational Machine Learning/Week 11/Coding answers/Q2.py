import numpy as np

def sigmoid(z):
    """
    Logistic sigmoid function sigma(z) = 1 / (1 + exp(-z)).
    
    Parameters
    ----------
    z : float or np.ndarray
        Input value or array of values.
        
    Returns
    -------
    np.ndarray
        Sigmoid of the input, with the same shape as `z`.
    """
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss_and_grad(w, X, y):
    """
    Compute the average logistic loss (binary cross-entropy) and its gradient.
    
    We assume a logistic regression model:
        p_i = sigmoid(w^T x_i)
    and loss:
        ell(w) = -1/N * sum_i [ y_i log(p_i) + (1 - y_i) log(1 - p_i) ].
    
    Parameters
    ----------
    w : np.ndarray of shape (M,)
        Parameter vector of the model.
    X : np.ndarray of shape (N, M)
        Design matrix; each row is a sample x_i.
    y : np.ndarray of shape (N,)
        Binary labels in {0, 1}.
        
    Returns
    -------
    loss : float
        The average logistic loss over the N samples.
    grad : np.ndarray of shape (M,)
        Gradient of the loss with respect to w.
    """
    # Number of samples.
    N = X.shape[0]
    
    # Compute linear scores z_i = w^T x_i for all samples at once.
    z = X @ w
    
    # Convert scores into probabilities p_i = sigma(z_i).
    p = sigmoid(z)  # shape (N,)
    
    # Clip probabilities to avoid log(0) when computing the loss.
    eps = 1e-12
    p_clipped = np.clip(p, eps, 1.0 - eps)
    
    # Compute average binary cross-entropy:
    #   loss = -1/N * sum_i [ y_i log(p_i) + (1-y_i) log(1-p_i) ].
    loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
    
    # Compute the gradient:
    #   grad = -1/N * sum_i (y_i - p_i) x_i.
    diff = y - p  # shape (N,)
    grad = -(1.0 / N) * (X.T @ diff)
    
    return loss, grad


if __name__ == "__main__":
    # Small random test to check shapes and that code runs.
    np.random.seed(0)
    N, M = 5, 3
    X_test = np.random.randn(N, M)
    y_test = (np.random.rand(N) > 0.5).astype(int)
    w_test = np.random.randn(M)
    
    loss_val, grad_val = logistic_loss_and_grad(w_test, X_test, y_test)
    print("Loss:", loss_val)
    print("Gradient:", grad_val)