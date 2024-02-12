import numpy as np


def l2_norm(x):
    if x.ndim == 1:
        return x / np.sqrt(np.sum(np.square(x)))
    else:
        raise ValueError

  
def min_max_norm(x):
    if x.ndim == 1:
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    else:
        raise ValueError

   
def mean_std_norm(x):
    if x.ndim == 1:
        return (x - np.mean(x)) / np.std(x)
    else:
        raise ValueError