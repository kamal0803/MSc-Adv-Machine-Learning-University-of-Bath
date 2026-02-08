import numpy as np

def sigmoid(z):
    """
    Logistic sigmoid function sigma(z) = 1 / (1 + exp(-z)).
    """
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss_and_grad(w, X, y):
    """
    Compute average logistic loss and its gradient.
    """
    N = X.shape[0]
    z = X @ w
    p = sigmoid(z)
    
    eps = 1e-12
    p_clipped = np.clip(p, eps, 1.0 - eps)
    
    loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
    diff = y - p
    grad = -(1.0 / N) * (X.T @ diff)
    
    return loss, grad


def train_logistic_regression(X, y, w_init, alpha=0.1, num_iters=1000):
    """
    Train logistic regression via batch gradient descent.
    
    Parameters
    ----------
    X : np.ndarray of shape (N, M)
        Training data matrix.
    y : np.ndarray of shape (N,)
        Binary labels in {0, 1}.
    w_init : np.ndarray of shape (M,)
        Initial parameter vector.
    alpha : float
        Learning rate.
    num_iters : int
        Number of gradient descent iterations.
        
    Returns
    -------
    w : np.ndarray of shape (M,)
        Learned parameter vector.
    loss_history : list of float
        Loss at each iteration.
    """
    w = w_init.copy()
    loss_history = []
    
    for t in range(num_iters):
        loss, grad = logistic_loss_and_grad(w, X, y)
        loss_history.append(loss)
        w = w - alpha * grad
        
        if (t + 1) % 100 == 0:
            print(f"Iteration {t+1:4d}/{num_iters}, loss = {loss:.4f}")
    
    return w, loss_history


if __name__ == "__main__":
    # Simple test run with random data.
    np.random.seed(1)
    N, M = 50, 3
    X_train = np.random.randn(N, M)
    y_train = (np.random.rand(N) > 0.5).astype(int)
    w0 = np.zeros(M)
    
    w_learned, losses = train_logistic_regression(X_train, y_train, w0,
                                                  alpha=0.1, num_iters=300)
    print("Final loss:", losses[-1])
    print("Learned weights:", w_learned)