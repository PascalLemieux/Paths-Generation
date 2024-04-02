import numpy as np


class RandomNumberGenerator1d(object):

    def __init__(self):
        raise NotImplementedError("Static class.")

    @classmethod
    def uniform(cls, samples: int, paths: int = 1):
        return np.random.random((samples, paths))

    @classmethod
    def normal(cls, samples: int, paths: int = 1, mean: float = 0.0, std: float = 1.0):
        return np.random.normal(mean, std, size=(samples, paths))
