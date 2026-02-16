import numpy as np

# -----------------------------
# Log unnormalised target density
# -----------------------------
def log_p(theta: np.ndarray) -> np.ndarray:  # Define log of the unnormalised density (up to a constant).
    """Return log p(theta) up to an additive constant.""" 
    theta = np.asarray(theta, dtype=float)  # Ensure theta is a float array for safe vector operations.
    term_quadratic = -0.5 * ((theta - 2.0) ** 2)  # Compute -(theta-2)^2 / 2 term.
    term_quartic = -0.1 * (theta ** 4)  # Compute -0.1 * theta^4 term (adds heavy tails penalty).
    return term_quadratic + term_quartic  # Sum terms to produce log density (unnormalised).

# -----------------------------
# Numerical second derivative (central difference)
# -----------------------------
def second_derivative(f, x0: float, eps: float = 1e-4) -> float:  # Define a helper to approximate f''(x0).
    """Approximate second derivative f''(x0) by central difference.""" 
    f_plus = float(f(x0 + eps))  # Evaluate f at x0 + eps.
    f_0 = float(f(x0))  # Evaluate f at x0.
    f_minus = float(f(x0 - eps))  # Evaluate f at x0 - eps.
    return (f_plus - 2.0 * f_0 + f_minus) / (eps ** 2)  # Apply central difference formula for second derivative.

# -----------------------------
# (a) MAP by grid search
# -----------------------------
theta_grid = np.linspace(-4.0, 4.0, 20001)  # Create a dense grid for searching the maximum.
log_vals = log_p(theta_grid)  # Compute log density values across the grid.
idx_max = int(np.argmax(log_vals))  # Find the index where log density is maximised.
theta_map = float(theta_grid[idx_max])  # Read off theta* (MAP) from the grid.
print(f"MAP estimate theta* = {theta_map:.6f}")  # Print MAP estimate.

# -----------------------------
# (b) Laplace approximation: local Gaussian at the MAP
# -----------------------------
logp_second = second_derivative(lambda t: log_p(np.array([t]))[0], theta_map)  # Estimate d^2/dtheta^2 log p at theta*.
neg_hessian = -logp_second  # Compute negative Hessian (in 1D: -logp'') at the mode.
sigma2 = 1.0 / neg_hessian  # Laplace variance approximation: sigma^2 = 1 / (-logp'').
sigma = float(np.sqrt(sigma2))  # Convert variance to standard deviation.
print(f"Laplace approx: mu={theta_map:.6f}, sigma^2={sigma2:.6f}, sigma={sigma:.6f}")  # Print Gaussian parameters.

# -----------------------------
# Optional: numeric sanity check by normalising both densities on the grid
# -----------------------------
p_unnorm = np.exp(log_vals - np.max(log_vals))  # Compute unnormalised p(theta) stably (subtract max log to prevent overflow).

# NOTE: Use np.trapezoid for numerical integration (NumPy >= 2.0).
p_area = np.trapezoid(p_unnorm, theta_grid)  # Approximate integral of unnormalised density over the grid.
p_norm = p_unnorm / p_area  # Normalise p(theta) to integrate to ~1 on the grid.

gauss_const = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)  # Compute normalisation constant for Gaussian N(mu, sigma^2).
q = gauss_const * np.exp(-0.5 * ((theta_grid - theta_map) / sigma) ** 2)  # Evaluate Gaussian approximation on the grid.

# Check integrals are close to 1 using trapezoid rule.
print(f"Integral target p(theta) ~ {np.trapezoid(p_norm, theta_grid):.6f}")  # Print integral of normalised target (~1).
print(f"Integral Laplace q(theta) ~ {np.trapezoid(q, theta_grid):.6f}")  # Print integral of Gaussian (~1).
