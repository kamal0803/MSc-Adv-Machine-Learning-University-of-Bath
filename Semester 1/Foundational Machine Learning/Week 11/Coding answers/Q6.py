import numpy as np
import matplotlib.pyplot as plt

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
    """
    w = w_init.copy()
    for _ in range(num_iters):
        loss, grad = logistic_loss_and_grad(w, X, y)
        w = w - alpha * grad
    return w


def plot_decision_boundary(X_raw, y, w):
    """
    Plot 2D data points and the decision boundary.
    
    Parameters
    ----------
    X_raw : np.ndarray of shape (N, 2)
        Original 2D features (without bias column).
    y : np.ndarray of shape (N,)
        Labels in {0, 1}.
    w : np.ndarray of shape (3,)
        Parameters [bias, w1, w2].
    """
    class0 = (y == 0)
    class1 = (y == 1)
    
    plt.figure(figsize=(6, 5))
    plt.scatter(X_raw[class0, 0], X_raw[class0, 1],
                marker='o', alpha=0.7, label='Class 0')
    plt.scatter(X_raw[class1, 0], X_raw[class1, 1],
                marker='s', alpha=0.7, label='Class 1')
    
    b, w1, w2 = w
    x1_min, x1_max = X_raw[:, 0].min() - 1.0, X_raw[:, 0].max() + 1.0
    x1_vals = np.linspace(x1_min, x1_max, 200)
    
    if np.abs(w2) < 1e-8:
        x1_boundary = -b / w1
        plt.axvline(x=x1_boundary, linestyle='--', label='Decision boundary')
    else:
        x2_vals = -(b + w1 * x1_vals) / w2
        plt.plot(x1_vals, x2_vals, 'k--', label='Decision boundary')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Logistic Regression Decision Boundary')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Generate synthetic 2D data.
    np.random.seed(123)
    N = 200
    X_raw = np.random.randn(N, 2)
    X = np.hstack([np.ones((N, 1)), X_raw])
    
    # True parameters (for data generation).
    w_true = np.array([-0.5, 1.5, -1.0])
    p_true = sigmoid(X @ w_true)
    y = (np.random.rand(N) < p_true).astype(int)
    
    # Train logistic regression.
    w_init = np.zeros(3)
    w_learned = train_logistic_regression(X, y, w_init,
                                          alpha=0.1, num_iters=1000)
    
    print("True weights:    ", w_true)
    print("Learned weights: ", w_learned)
    
    # Plot the data and learned decision boundary.
    plot_decision_boundary(X_raw, y, w_learned)