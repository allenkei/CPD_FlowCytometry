import numpy as np
import pandas as pd
import os

import numpy as np

def generate_x(T=200, N=100, dx=3):
    """
    Generate covariates X with structured temporal patterns:
      - Every 3 dimensions form one group:
        (1) increasing,
        (2) decreasing,
        (3) wavelet (sinusoidal)
    Each group is repeated floor(dx/3) times, extra dims are random normal.

    Parameters
    ----------
    T : int
        Number of time steps
    N : int
        Number of samples per time
    dx : int
        Dimension of covariates

    Returns
    -------
    X : np.ndarray of shape (T, N, dx)
    """
    t = np.linspace(0, 1, T)
    n_groups = dx // 3
    X = np.zeros((T, N, dx))

    for g in range(n_groups):
        # (1) increasing
        inc = t[:, None] + 0.1 * np.random.randn(T, N)
        # (2) decreasing
        dec = (1 - t)[:, None] + 0.1 * np.random.randn(T, N)
        # (3) wavelet (simple sine wave)
        wave = np.sin(4 * np.pi * t)[:, None] + 0.1 * np.random.randn(T, N)

        X[:, :, 3*g + 0] = inc
        X[:, :, 3*g + 1] = dec
        X[:, :, 3*g + 2] = wave

    # If dx not multiple of 3, fill remaining dims with random noise
    rem = dx % 3
    if rem > 0:
        X[:, :, -rem:] = np.random.randn(T, N, rem)

    return X



def generate_data(X, mean_func, sigma_func, change_points=[100], z_means=[0, 5]):
    """
    Generate time series data with latent Z and multiple change points.

    Parameters
    ----------
    X : np.ndarray
        Covariates, shape (T, N, dx)
    mean_func : callable
        Function mean_func(h, z) -> (dy,)
    sigma_func : callable
        Function sigma_func(h, z) -> (dy,)
    change_points : list of int
        Time indices where Z distribution changes
    z_means : list of float
        Mean of Z distribution in each segment. 
        Length must be len(change_points)+1.

    Returns
    -------
    X : np.ndarray (T, N, dx)
    Y : np.ndarray (T, N, dy)
    Z : np.ndarray (T, N, dz)
    """
    T, N, dx = X.shape
    dz = 3
    test_h = np.concatenate([X[0,0], np.zeros(dz)])
    dy = mean_func(test_h, np.zeros(dz)).shape[0]

    if len(z_means) != len(change_points) + 1:
        raise ValueError("len(z_means) must equal len(change_points)+1")

    Z = np.zeros((T, N, dz))
    Y = np.zeros((T, N, dy))

    cps = [0] + change_points + [T]

    for seg in range(len(cps)-1):
        mu_z = z_means[seg]
        for t in range(cps[seg], cps[seg+1]):
            for n in range(N):
                z = np.random.normal(mu_z, 1, dz)
                Z[t,n] = z
                h = np.concatenate([X[t,n], z])
                mu = mean_func(h, z)
                sigma = sigma_func(h, z)
                Y[t,n] = np.random.normal(mu, sigma)

    return X, Y, Z


u = np.random.randn(6)+1  # dx + dz = 3 + 3 = 6
v = np.random.randn(6)+2
w = np.random.randn(6)-1
d = np.random.randn(6)-2

def mean_func(h, z):
    return np.array([
        h @ u + np.tanh(h @ u) + z[0],
        h @ v + np.sin(h @ v) + z[1],
        h @ w + 0.5*(h @ w) - z[2]
    ])

def sigma_func(h, z):
    val = 0.1 * abs(h @ d) + 0.01 * abs(z[0])
    sigma = 0.1 + 0.1 * np.tanh(val) + abs(z[0]) * 0.5
    return sigma * np.ones(3)  # dy = 3