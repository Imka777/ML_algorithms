import numpy as np


def euclidean_distance(x, y):
    x_norm = np.sum(x**2, axis=1)
    y_norm = np.sum(y**2, axis=1)
    xy = x @ y.T
    
    return np.sqrt(x_norm.reshape(-1, 1) - 2*xy + y_norm)


def cosine_distance(x, y):
    x_norm = np.sum(x**2, axis=1)
    y_norm = np.sum(y**2, axis=1)
    xy = x @ y.T

    return 1 - xy / np.sqrt(x_norm.reshape(-1, 1)) / np.sqrt(y_norm) 
