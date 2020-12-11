import numpy as np


def moving_average(a, window=10):
    '''
    Computes the moving average
    '''
    ret = np.cumsum(a, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window - 1:] / window


def epsilon_decay(iteration: int) -> float:
    if iteration <= 1000000:
        return 1 - 0.9 * iteration / 1000000
    return 0.1

