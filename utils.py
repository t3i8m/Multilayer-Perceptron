import numpy as np


def sigmoid_function(x:np.array)->np.array:
    """A sigmoid activation function implementation"""
    return 1.0/(1.0+ np.exp(-x))