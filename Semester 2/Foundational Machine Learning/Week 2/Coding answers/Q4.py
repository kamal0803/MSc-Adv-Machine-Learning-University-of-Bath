import numpy as np
import time  # Import time module for performance timing.

# -----------------------------
# Gaussian kernel + KDE evaluator
# -----------------------------
def gaussian_kernel(u: np.ndarray) -> np.ndarray:  # Define Gaussian kernel function.
    """Standard normal density kernel."""
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (u ** 2))  # Compute kernel values.

def gaussian_kde(x: np.ndarray, data: np.ndarray, h: float) -> np.ndarray:  # Define KDE evaluation routine.
    """Vectorised Gaussian KDE across all query points x."""
    x = np.asarray(x, dtype=float)  # Ensure queries are float array.
    data = np.asarray(data, dtype=float)  # Ensure samples are float array.
    N = data.size  # Store N for normalisation.
    u = (x[:, None] - data[None, :]) / h  # Build (T, N) matrix of scaled distances.
    K = gaussian_kernel(u)  # Evaluate kernel contributions for each (query, sample) pair.
    return (1.0 / (N * h)) * np.sum(K, axis=1)  # Sum over samples for each query and normalise by Nh.

# -----------------------------
# Experiment setup
# -----------------------------
rng = np.random.default_rng(0)  # Create reproducible random generator.
h = 1.0  # Fix bandwidth to isolate scaling effects.

Ns = [200, 1000, 5000, 20000]  # Define dataset sizes N to test.
Ts = [50, 200, 800]  # Define query sizes T to test.

# -----------------------------
# Run timing loops
# -----------------------------
for N in Ns:  # Loop over dataset sizes.
    data = rng.normal(loc=0.0, scale=1.0, size=N)  # Generate N samples from N(0,1).
    for T in Ts:  # Loop over query counts.
        x = np.linspace(-3.0, 3.0, T)  # Create T evenly spaced queries.
        start = time.perf_counter()  # Start high-resolution timer.
        _ = gaussian_kde(x, data, h)  # Run KDE (discard output; we only time computation).
        end = time.perf_counter()  # Stop timer.
        elapsed_ms = (end - start) * 1000.0  # Convert elapsed time to milliseconds.
        print(f"N={N:>6d}, T={T:>4d} -> time={elapsed_ms:>8.3f} ms")  # Print timing measurement.