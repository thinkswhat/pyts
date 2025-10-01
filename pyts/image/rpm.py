"""Code for Relative Position Matrix."""

# Author: Lucky J. Yang <jianqiy4@gmail.com>
# License: BSD-3-Clause

import numpy as np

def RelativePositionMatrix(x, k):
    """
    Compute Relative Position Matrix (RPM) of a 1D time series
    
    Parameters:
        x : array-like, 1D time series
        k : int, PAA reduction factor
        visualize : bool, whether to display RPM as image
        
    Returns:
        RPM : np.ndarray, relative position matrix scaled to [0,255]
    """
    x = np.array(x, dtype=float).flatten()
    
    # Standardize
    mu = np.mean(x)
    delta = np.sqrt(np.var(x))
    z = (x - mu) / delta
    
    # PAA
    N = len(z)
    m = int(np.ceil(N / k))
    X = np.zeros(m)
    
    if np.ceil(N / k) - np.floor(N / k) == 0:  # N divisible by k
        for i in range(m):
            X[i] = np.mean(z[i*k:(i+1)*k])
    else:
        for i in range(m-1):
            X[i] = np.mean(z[i*k:(i+1)*k])
        X[m-1] = np.mean(z[(m-1)*k:N])
    
    # Relative position matrix
    M = X[:, None] - X[None, :]
    
    # Scale to [0, 255]
    min_val = M.min()
    max_val = M.max()
    RPM = (M - min_val) / (max_val - min_val) * 255
        
    return RPM
