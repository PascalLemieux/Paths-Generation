import six
import numpy as np
import pandas as pd

from src.generators.samples import RandomNumberGenerator1d
from src.models.base import StochasticProcess


class BrownianBridge(StochasticProcess):
    """
    Class to generate trajectories for a Brownian Bridge.

    To use, instantiate the process with a given set of characteristics,
    then just call on the instance with the desired number of paths.

    For example, to generate 1,000 paths:

    bb = BrownianBridge(volatility=0.25)
    paths = bb(1000)

    """

    def __init__(self, volatility: float, maturity: float = 1.0, time_intervals: int = 365):
        #
        # Parent
        super().__init__(volatility=volatility, initial_value=0.0,
                         maturity=maturity, time_intervals=time_intervals)

    def __generate_bridge(self, nb_paths: int):

        # Set for future reference
        self.nb_paths = nb_paths

        # References for speed...
        time_steps = self.time.steps
        vol = self.volatility

        # Time scaling
        dt = 1.0 / (time_steps - 1)
        dt_sqrt = np.sqrt(dt)

        bridge = np.empty((nb_paths, time_steps), dtype=np.float32)
        bridge[:, 0] = 0

        for n in range(0, time_steps - 2):
            t = n * dt
            xi = np.random.randn(nb_paths) * dt_sqrt * vol
            bridge[:, n + 1] = bridge[:, n] * (1 - dt / (1 - t)) + xi

        # Force to zero on last time step
        bridge[:, -1] = 0.0

        return pd.DataFrame(bridge.T, index=self.schedule)

    def generate(self, nb_paths: int = 1, regenerate: bool = True):
        if self.paths is None or regenerate or self.nb_paths != nb_paths:
            self.paths = self.__generate_bridge(nb_paths=nb_paths)
        return self.paths


if __name__ == '__main__':
    b_bridge = BrownianBridge(volatility=0.25)
    b_bridge(100)
