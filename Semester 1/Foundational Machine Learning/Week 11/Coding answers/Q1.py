import numpy as np

def sigmoid(z):
    """
    Compute the logistic sigmoid function sigma(z) = 1 / (1 + exp(-z)).
    
    Parameters
    ----------
    z : float or np.ndarray
        Input value or array of values.
        
    Returns
    -------
    np.ndarray
        Sigmoid of the input, with the same shape as `z`.
    """
    # Use the standard definition: sigma(z) = 1 / (1 + exp(-z)).
    # NumPy will automatically apply exp element-wise if z is an array.
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    """
    Compute the logit function logit(p) = log(p / (1 - p)).
    
    Parameters
    ----------
    p : float or np.ndarray
        Probabilities in the open interval (0, 1).
        
    Returns
    -------
    np.ndarray
        Logit of the input probabilities.
    """
    # Convert input to a NumPy array for convenience in vectorised operations.
    p = np.asarray(p)
    
    # Basic input check:
    # We require probabilities to be strictly between 0 and 1 to avoid log(0).
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("All probabilities must be strictly between 0 and 1.")
    
    # Apply the logit transform element-wise.
    return np.log(p / (1.0 - p))


# ---- Simple test: logit(sigmoid(z)) \approx z ----

# Create a range of test values for z.
z_values = np.linspace(-5, 5, num=11)  # 11 points from -5 to 5

# Apply sigmoid, then logit.
sig_values = sigmoid(z_values)
reconstructed_z = logit(sig_values)

print("Original z values:     ", z_values)
print("Reconstructed z values:", reconstructed_z)
print("Difference:            ", reconstructed_z - z_values)