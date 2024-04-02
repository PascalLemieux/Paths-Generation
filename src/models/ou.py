import numpy as np
import pandas as pd

from src.generators.samples import RandomNumberGenerator1d
from src.models.base import StochasticProcess


class OrnsteinUhlenbeckProcess(StochasticProcess):
    """
    Class to generate trajectories from a Ornstein Uhlenbeck process.

    To use, instantiate the process with a given set of characteristics,
    then just call on the instance with the desired number of paths.

    For example, to generate 100 paths:

    ou = OrnsteinUhlenbeckProcess(volatility=0.01)
    paths = ou(100)

    """

    def __init__(self, volatility: float, long_term_mean: float = 1.0, mean_reversion: float = 0.0,
                 initial_value: float = 1.0, maturity: float = 1.0, time_intervals: int = 365):
        super().__init__(volatility=volatility, initial_value=initial_value,
                         maturity=maturity, time_intervals=time_intervals)

        # Process Characteristics
        self.long_term_mean = long_term_mean  # annual
        self.mean_reversion = mean_reversion  # annual

    @property
    def effective_std_dev(self):
        elasticity = 0.5 / self.mean_reversion * (1.0 - np.exp(-2.0 * self.mean_reversion * self.time.dt))
        return np.sqrt(elasticity * self.volatility ** 2.0)

    def __generate_ou(self, nb_paths):

        # Reference these...
        nb_time_steps = self.time.steps
        kappa = self.mean_reversion
        dt = self.time.dt
        theta = self.long_term_mean

        # Random normal generation
        # epsilon = np.random.normal(0, 1, (nb_paths, nb_time_steps - 1))
        epsilon = RandomNumberGenerator1d.normal(samples=nb_time_steps-1, paths=nb_paths).T

        # Initialization
        _ou = np.zeros((nb_paths, nb_time_steps))
        _ou[:, 0] = self.initial_value

        # Randomness
        _ou[:, 1:] = np.kron(self.effective_std_dev, np.ones((nb_paths, 1))) * epsilon

        for i in range(1, _ou.shape[1]):
            _ou[:, i] += theta * (1 - np.exp(-kappa * dt))
            _ou[:, i] += np.exp(-kappa * dt) * _ou[:, i - 1]

        return pd.DataFrame(np.transpose(_ou), index=self.schedule)

    def generate(self, nb_paths: int, regenerate: bool = True):
        if self.paths is None or regenerate or self.nb_paths != nb_paths:
            self.paths = self.__generate_ou(nb_paths=nb_paths)
        return self.paths

    def theoretical_expectation(self, spot_0: float = None, maturity: float = None,
                                mr: float = None, lt_mean: float = None):
        spot_0 = self.initial_value if spot_0 is None else spot_0
        maturity = self.time.maturity if maturity is None else maturity
        mr = self.mean_reversion if mr is None else mr
        lt_mean = self.long_term_mean if lt_mean is None else lt_mean
        return spot_0 * np.exp(-1.0 * mr * maturity) + lt_mean * (1.0 - np.exp(-1.0 * mr * maturity))

    def theoretical_variance(self, maturity: float = None, sigma: float = None, mr: float = None):
        maturity = self.time.maturity if maturity is None else maturity
        mr = self.mean_reversion if mr is None else mr
        sigma = self.volatility if sigma is None else sigma
        return 0.5 * sigma ** 2 * (1.0 - np.exp(-2.0 * mr * maturity)) / mr

    def theoretical_std_dev(self, maturity: float = None, sigma: float = None, mr: float = None):
        return np.sqrt(self.theoretical_variance(maturity, sigma, mr))


if __name__ == '__main__':
    ou = OrnsteinUhlenbeckProcess(volatility=0.25)
    ou.generate(100)
