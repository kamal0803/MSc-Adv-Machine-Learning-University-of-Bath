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
    """
    w = w_init.copy()
    loss_history = []
    
    for t in range(num_iters):
        loss, grad = logistic_loss_and_grad(w, X, y)
        loss_history.append(loss)
        w = w - alpha * grad
        
    return w, loss_history


def predict_proba(X, w):
    """
    Predict class-1 probabilities p(y=1 | x).
    """
    return sigmoid(X @ w)


def predict_labels(X, w, threshold=0.5):
    """
    Predict class labels using a probability threshold.
    """
    p = predict_proba(X, w)
    return np.where(p >= threshold, 1, 0)


def accuracy(y_true, y_pred):
    """
    Classification accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    # 1) Generate synthetic data.
    np.random.seed(42)
    N = 200  # number of samples
    
    # True parameters, including bias: w_true = [b, w1, w2]
    w_true = np.array([-0.5, 2.0, -1.0])
    
    # Generate raw 2D features.
    X_raw = np.random.randn(N, 2)  # shape (N, 2)
    
    # Add bias column of ones: X has shape (N, 3) = [1, x1, x2]
    X = np.hstack([np.ones((N, 1)), X_raw])
    
    # Compute probabilities under the true model and sample labels.
    z_true = X @ w_true
    p_true = sigmoid(z_true)
    y = (np.random.rand(N) < p_true).astype(int)
    
    # 2) Train logistic regression.
    w_init = np.zeros(X.shape[1])
    alpha = 0.1
    num_iters = 1000
    
    w_learned, loss_history = train_logistic_regression(
        X, y, w_init, alpha=alpha, num_iters=num_iters
    )
    
    # 3) Evaluate training accuracy.
    y_pred = predict_labels(X, w_learned, threshold=0.5)
    train_acc = accuracy(y, y_pred)
    
    print("True weights:    ", w_true)
    print("Learned weights: ", w_learned)
    print(f"Training accuracy: {train_acc * 100:.2f} %")