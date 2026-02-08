import numpy as np

def sigmoid(z):
    """
    Logistic sigmoid function sigma(z) = 1 / (1 + exp(-z)).
    """
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X, w):
    """
    Predict class-1 probabilities p(y=1 | x) for all samples in X.
    
    Parameters
    ----------
    X : np.ndarray of shape (N, M)
        Data matrix.
    w : np.ndarray of shape (M,)
        Parameter vector.
        
    Returns
    -------
    p : np.ndarray of shape (N,)
        Predicted probabilities for class 1.
    """
    z = X @ w
    p = sigmoid(z)
    return p


def predict_labels(X, w, threshold=0.5):
    """
    Predict binary labels in {0, 1} for all samples in X.
    
    Parameters
    ----------
    X : np.ndarray of shape (N, M)
        Data matrix.
    w : np.ndarray of shape (M,)
        Parameter vector.
    threshold : float
        Decision threshold; if p >= threshold, predict 1, else 0.
        
    Returns
    -------
    y_pred : np.ndarray of shape (N,)
        Predicted labels.
    """
    p = predict_proba(X, w)
    y_pred = np.where(p >= threshold, 1, 0)
    return y_pred


def accuracy(y_true, y_pred):
    """
    Compute classification accuracy: fraction of correct predictions.
    
    Parameters
    ----------
    y_true : np.ndarray of shape (N,)
        True labels.
    y_pred : np.ndarray of shape (N,)
        Predicted labels.
        
    Returns
    -------
    acc : float
        Accuracy in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    # Small example: random scores and labels.
    np.random.seed(2)
    N, M = 10, 3
    X_demo = np.random.randn(N, M)
    w_demo = np.random.randn(M)
    y_true_demo = (np.random.rand(N) > 0.5).astype(int)
    
    y_pred_demo = predict_labels(X_demo, w_demo, threshold=0.5)
    acc_demo = accuracy(y_true_demo, y_pred_demo)
    
    print("True labels:     ", y_true_demo)
    print("Predicted labels:", y_pred_demo)
    print("Accuracy:        ", acc_demo)