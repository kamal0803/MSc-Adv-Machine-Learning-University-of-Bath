import numpy as np 

# -----------------------------
# Data and parameters
# -----------------------------
samples = np.array([-2.1, -1.3, -0.4, 1.9, 5.1, 6.2], dtype=float)  # Store the given samples x_i as a float array.
N = samples.size  # Store the number of samples N (used to normalise counts into a density).
h = 1.0  # Set bandwidth h = 1 because bin width is 2h = 2.

query_points = np.array([-2.0, 0.0, 2.0, 4.0, 6.0], dtype=float)  # Define the query x values where we estimate f(x).

# -----------------------------
# Interval-counting estimator
# -----------------------------
def interval_density_estimate(x: float, data: np.ndarray, h: float) -> float:  # Define a function that computes the estimator at one x.
    """Compute interval-counting density estimate at query x."""
    left = x - h  # Compute left endpoint of the counting window [x-h, x+h].
    right = x + h  # Compute right endpoint of the counting window [x-h, x+h].
    in_interval = (data >= left) & (data <= right)  # Build boolean mask: True for points falling inside [left, right].
    count = int(np.sum(in_interval))  # Count how many samples are inside the window (# of True values).
    density = (1.0 / (2.0 * h)) * (count / data.size)  # Apply formula: (1/(2h)) * (count / N).
    return density  # Return the density estimate.

# -----------------------------
# Compute and print results
# -----------------------------
for x in query_points:  # Loop over each query point.
    fx = interval_density_estimate(float(x), samples, h)  # Compute f_hat(x) using the estimator function.
    print(f"x={x:>4.1f}  f_hat={fx:.6f}")  # Print a formatted line with x and its estimated density.