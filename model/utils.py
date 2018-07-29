import numpy as np

def sample_z(num=128, dim=100):
    return np.random.normal(size=(num, dim))