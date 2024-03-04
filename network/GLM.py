import numpy as np
from scipy.optimize import minimize

def poisson_log_likelihood(W_flat, X, phi, b, N, K, sigma):
    """
    Computes the negative Poisson log-likelihood of the observed spikes given the weight matrix W.

    Parameters:
    - W_flat: The flattened weight matrix W of shape (N*N,).
    - X: The spike data matrix of shape (T, N).
    - phi: The basis functions of shape (K,).
    - b: The background intensity vector of shape (N,).
    - N: The number of neurons.
    - K: The number of basis functions.
    - sigma: The non-linear function.

    Returns:
    - The negative log-likelihood to be minimized.
    """
    T = X.shape[0]
    W = W_flat.reshape(N, N)
    log_likelihood = 0

    # Pre-compute the weighted sum of spikes for each basis function
    weighted_spikes = np.zeros((T, N))
    for k in range(K):
        X_shifted = np.roll(X, k + 1, axis=0)  # Shift X by k+1 to align with the basis function timing
        X_shifted[:k + 1, :] = 0  # Zero out the shifted-in values at the start
        weighted_spikes += phi[k] * X_shifted @ W  # Matrix multiplication for weighted sum

    # Compute the linear predictor for all time points and neurons
    linear_predictor = b + weighted_spikes

    # Apply the non-linear function (e.g., Softplus) and calculate the log-likelihood
    rates = sigma(linear_predictor)
    log_likelihood = np.sum(X * np.log(rates) - rates)


    # for t in range(T):
    #     for n in range(N):
    #         linear_predictor = b[n]
    #         for n_prime in range(N):
    #             for k in range(K):
    #                 if t - k - 1 >= 0:
    #                     linear_predictor += W[n, n_prime] * X[t - k - 1, n_prime] * phi[k]
    #         rate = sigma(linear_predictor)
    #         log_likelihood += X[t, n] * np.log(rate) - rate  # Poisson log-likelihood component

    return -log_likelihood  # Minimize the negative log-likelihood



def estimate_W_default_params():
    # Example usage
    T, N, K = 50, 10, 2  # time bins, neurons, basis functions
    X = np.random.poisson(5, (T, N))  # Simulated spike data
    print(X)
    phi = np.linspace(0, 1, K)  # Example basis functions
    b = np.random.normal(0, 1, N)  # Background intensities
    W = np.random.normal(0, 1, (N, N))  # Weight matrix

    # Non-linear function, using the sigmoid function as an example
    sigma = lambda x: 1 / (1 + np.exp(-x))

    W_est = estimate_W(N, X, phi, b, K, sigma)
    print(W_est)


def estimate_W(N, X, phi=None, b=None, K=2, sigma=lambda x: np.log1p(np.exp(x))):
    if phi is None:
        phi = np.linspace(0, 1, K)
    if b is None:
        b = np.random.normal(0, 1, N)

    result = minimize(
        poisson_log_likelihood,
        x0=np.random.normal(0, 1, N * N),  # Initial guess for W
        args=(X, phi, b, N, K, sigma),
        method='L-BFGS-B'  # Optimization method suitable for large problems
    )

    W_estimated = result.x.reshape(N, N)
    #print('Estimated shape', W_estimated.shape)  # Should be (N, N)
    return W_estimated

