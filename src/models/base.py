from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.generators.time import TimeGenerator


class StochasticProcess(ABC):

    def __init__(self, volatility: float, maturity: float = 1.0, time_intervals: int = 365,
                 initial_value: float = 1.0, seed: int = 0):
        #
        # Common to all stochastic processes
        self.volatility = max(0.0, float(volatility))
        self.initial_value = float(initial_value)  # spot at t=0

        # Time schedule
        self.time = TimeGenerator(maturity=maturity, intervals=time_intervals)

        # Monte Carlo paths
        self.paths: Optional[pd.DataFrame] = None
        self.nb_paths: int = 0

        self.seed = seed if seed is not None else 0
        np.random.seed(self.seed)

    @property
    def schedule(self):
        return self.time()

    @property
    def dt(self):
        return self.time.dt

    def __call__(self, nb_paths: int, regenerate: bool = True):
        return self.generate(nb_paths, regenerate=regenerate)

    @abstractmethod
    def generate(self, nb_paths: int = 1, regenerate: bool = True):
        raise NotImplementedError("Child classes must implement this method.")

    def to_datetime_index(self, start: datetime = None, basis=365):
        if self.paths is None:
            self.generate()
        start = start if start is not None else datetime.today()
        idx_dt = [start + timedelta(days=i*basis) for i in self.paths.index]
        return pd.DataFrame(self.paths.to_numpy(), index=idx_dt)

