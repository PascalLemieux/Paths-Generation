import numpy as np
import pandas as pd

from src.generators.samples import RandomNumberGenerator1d
from src.generators.time import TimeGenerator
from src.models.base import StochasticProcess


class GeometricBrownianMotion(StochasticProcess):

    def __init__(self, volatility: float, drift: float = 0.0, initial_value: float = 1.0,
                 maturity: float = 1.0, time_intervals: int = 365):
        super().__init__(volatility=volatility, initial_value=initial_value,
                         maturity=maturity, time_intervals=time_intervals)

        # GBM Characteristic
        self.drift = float(drift)  # annual

    @property
    def dW(self):
        _dW = RandomNumberGenerator1d.normal(samples=self.time.intervals + 1, paths=self.nb_paths)
        _dW[0, :] = 0.0  # no uncertainty at present
        return _dW

    @property
    def sigma_dW(self):
        return self.volatility * np.sqrt(self.time.dt) * self.dW

    @property
    def r_dt(self):
        drift_per_period = np.power(1.0 + self.drift, self.time.dt) - 1.0
        _r_dt = np.ones(shape=(self.time.steps, self.nb_paths)) * drift_per_period
        _r_dt[0, :] = 0.0
        return _r_dt

    @property
    def ito_drift(self):
        # assumes dW is 0.0 on step 0
        return 0.5 * np.power(self.sigma_dW, 2)

    def __generate_gbm(self, nb_paths):
        self.nb_paths = max(1, int(nb_paths))
        log_s = np.exp(self.r_dt + self.sigma_dW - self.ito_drift)
        base_1 = np.cumprod(log_s, axis=0)
        return pd.DataFrame(base_1 * self.initial_value, index=self.schedule)

    def generate(self, nb_paths: int, regenerate: bool = True):
        if self.paths is None or regenerate or self.nb_paths != nb_paths:
            self.paths = self.__generate_gbm(nb_paths=nb_paths)
        return self.paths

    def __call__(self, nb_paths, regenerate: bool = True):
        return self.generate(nb_paths=nb_paths, regenerate=regenerate)

    def theoretical_expectation(self, spot_0: float = None, maturity: float = None, drift: float = None):
        spot_0 = self.initial_value if spot_0 is None else spot_0
        maturity = self.time.maturity if maturity is None else maturity
        drift = self.drift if drift is None else drift
        return spot_0 * np.exp(drift * maturity)

    def theoretical_variance(self, spot_0: float = None, maturity: float = None,
                             drift: float = None, sigma: float = None):
        spot_0 = self.initial_value if spot_0 is None else spot_0
        maturity = self.time.maturity if maturity is None else maturity
        sigma = self.volatility if sigma is None else sigma
        drift = self.drift if drift is None else drift
        return spot_0 ** 2 * np.exp(2.0 * drift * maturity) * (np.exp(sigma ** 2 * maturity) - 1.0)

    def theoretical_std_dev(self, spot_0: float = None, maturity: float = None,
                            drift: float = None, sigma: float = None):
        return np.sqrt(self.theoretical_variance(spot_0, maturity, drift, sigma))


if __name__ == '__main__':
    gbm = GeometricBrownianMotion(volatility=0.25)
    gbm(100)
