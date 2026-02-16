import numpy as np 

# -----------------------------
# Data
# -----------------------------
samples = np.array([-2.1, -1.3, -0.4, 1.9, 5.1, 6.2], dtype=float)  # Store training samples.

# -----------------------------
# Gaussian kernel + KDE
# -----------------------------
def gaussian_kernel(u: np.ndarray) -> np.ndarray:  # Define the Gaussian kernel.
    """Standard normal density kernel."""
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (u ** 2))  # Compute standard normal PDF values.

def gaussian_kde(x: np.ndarray, data: np.ndarray, h: float) -> np.ndarray:  # Define Gaussian KDE evaluator.
    """Evaluate Gaussian KDE at query points x with bandwidth h."""
    x = np.asarray(x, dtype=float)  # Ensure x is an array for broadcasting.
    data = np.asarray(data, dtype=float)  # Ensure data is an array.
    N = data.size  # Sample count N.
    u = (x[:, None] - data[None, :]) / h  # Compute scaled distances for all query points and samples.
    K = gaussian_kernel(u)  # Evaluate kernel contributions at those scaled distances.
    return (1.0 / (N * h)) * np.sum(K, axis=1)  # Sum contributions over samples and normalise by Nh.

# -----------------------------
# Grid and bandwidths
# -----------------------------
x_grid = np.linspace(-4.0, 8.0, 400)  # Create a dense grid so the peak is captured accurately.
bandwidths = [0.3, 1.0, 2.5]  # Choose small/medium/large bandwidths to compare.

# -----------------------------
# Evaluate and summarise
# -----------------------------
for h in bandwidths:  # Loop over each bandwidth.
    f = gaussian_kde(x_grid, samples, h)  # Compute KDE values on the grid.
    peak = float(np.max(f))  # Compute the maximum value as a simple "spikiness" summary.
    print(f"h={h:>3.1f}  peak_density={peak:.6f}")  # Print bandwidth and corresponding peak height.