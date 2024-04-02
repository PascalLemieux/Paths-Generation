from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

from src.generators.time import TimeGenerator


class StochasticProcess(ABC):

    def __init__(self, volatility: float, maturity: float = 1.0, time_intervals: int = 365,
                 initial_value: float = 1.0):
        #
        # Common to all stochastic processes
        self.volatility = max(0.0, float(volatility))
        self.initial_value = float(initial_value)  # spot at t=0

        # Time schedule
        self.time = TimeGenerator(maturity=maturity, intervals=time_intervals)

        # Monte Carlo paths
        self.paths: Optional[pd.DataFrame] = None
        self.nb_paths: int = 0

    @property
    def schedule(self):
        return self.time()

    @property
    def dt(self):
        return self.time.dt

    def __call__(self, nb_paths: int, regenerate: bool = True):
        return self.generate(nb_paths, regenerate=regenerate)

    @abstractmethod
    def generate(self, nb_paths: int, regenerate: bool = True):
        raise NotImplementedError("Child classes must implement this method.")
