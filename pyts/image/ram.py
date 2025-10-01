"""Code for Relative Angle Matrix."""

# Author: Lucky J. Yang <jianqiy4@gmail.com>
# License: BSD-3-Clause

import numpy as np

def RelativeAngleMatrix(V, d):
    """
    V: Input sequence (1D array or list)
    d: Number of rows in the target matrix
    Returns:
        A: Difference matrix transformed to [0, 255]
    """
    V = np.array(V, dtype=float)
    
    # Standardize the input
    X = (V - np.mean(V)) / np.std(V)
    
    # Reshape sequence into matrix with d rows
    m = len(X)
    n = int(np.ceil(m / d))
    if m % d != 0:
        X = np.concatenate([X, X[:d*n - m]])
    #M = X.reshape(d, n)
    M = X.reshape(d, n, order='F')  # 列优先，与 MATLAB 对齐
    
    # Center vector of columns
    center_vector = np.mean(M, axis=1)
    
    # Compute cosine similarity vectorized
    M_norm = np.linalg.norm(M, axis=0)
    center_norm = np.linalg.norm(center_vector)
    cos_beta = (M.T @ center_vector) / (M_norm * center_norm)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)  # 防止数值误差
    beta = np.arccos(cos_beta)
    
    # Difference matrix (向量化)
    R = beta[:, None] - beta[None, :]
    
    # Transform to [0, 255]
    A = (R - R.min()) / (R.max() - R.min()) * 255
    
    return A
