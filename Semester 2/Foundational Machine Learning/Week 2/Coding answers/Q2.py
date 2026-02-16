import numpy as np 

# -----------------------------
# Data and parameter
# -----------------------------
samples = np.array([-2.1, -1.3, -0.4, 1.9, 5.1, 6.2], dtype=float)  # Store training samples x_i.
N = samples.size  # Number of samples N.
h = 1.0  # Bandwidth h (bin width 2h = 2).

# -----------------------------
# Kernel definitions
# -----------------------------
def box_kernel(u: np.ndarray) -> np.ndarray:  # Define the box kernel function G(u).
    """Box kernel: G(u)=1/2 for |u|<=1 else 0."""
    inside = np.abs(u) <= 1.0  # Check which u values lie within [-1, 1].
    return np.where(inside, 0.5, 0.0)  # Return 1/2 for inside, else 0 (vectorised).

def gaussian_kernel(u: np.ndarray) -> np.ndarray:  # Define the Gaussian kernel function G(u).
    """Gaussian kernel: standard normal density evaluated at u."""
    norm_const = 1.0 / np.sqrt(2.0 * np.pi)  # Compute normalisation constant 1/sqrt(2*pi).
    return norm_const * np.exp(-0.5 * (u ** 2))  # Compute phi(u) elementwise.

# -----------------------------
# General KDE evaluator
# -----------------------------
def kde_estimate(x: np.ndarray, data: np.ndarray, h: float, kernel_fn) -> np.ndarray:  # Define a general KDE evaluation routine.
    """Evaluate KDE on query points x using kernel_fn."""
    x = np.asarray(x, dtype=float)  # Ensure x is a float array (supports broadcasting).
    data = np.asarray(data, dtype=float)  # Ensure data is a float array.
    N = data.size  # Compute N for normalisation.
    u = (x[:, None] - data[None, :]) / h  # Build matrix of scaled distances u_{t,i} = (x_t - x_i)/h.
    K = kernel_fn(u)  # Apply kernel to all scaled distances (same shape as u).
    fx = (1.0 / (N * h)) * np.sum(K, axis=1)  # Sum contributions over samples, then multiply by 1/(Nh).
    return fx  # Return KDE values for each query point.

# -----------------------------
# Evaluate KDEs on a grid
# -----------------------------
x_grid = np.linspace(-4.0, 8.0, 49)  # Create a grid of 49 x values for evaluation.
f_box = kde_estimate(x_grid, samples, h, box_kernel)  # Compute box-kernel KDE on the grid.
f_gauss = kde_estimate(x_grid, samples, h, gaussian_kernel)  # Compute Gaussian-kernel KDE on the grid.

# -----------------------------
# Print a small comparison table
# -----------------------------
for i in range(0, x_grid.size, 8):  # Print every 8th grid point to keep output compact.
    x_val = x_grid[i]  # Extract the x value at index i.
    print(f"x={x_val:>6.2f}  box={f_box[i]:.6f}  gauss={f_gauss[i]:.6f}")  # Print both estimates side-by-side.