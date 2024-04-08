import numpy as np
import pandas as pd

from src.generators.samples import RandomNumberGenerator1d
from src.generators.time import TimeGenerator
from src.models.base import StochasticProcess


class GeometricBrownianMotion(StochasticProcess):
    """
    Class to generate trajectories from the Geo Brownian Motion process.

    To use, instantiate the process with a given set of characteristics,
    then just call on the instance with the desired number of paths.

    For example, to generate 1,000 paths:

    gbm = GeometricBrownianMotion(volatility=0.25)
    paths = gbm(1000)

    """

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
        drift_per_period = np.exp(self.drift * self.time.dt) - 1.0
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

    def generate(self, nb_paths: int = 1, regenerate: bool = True):
        if self.paths is None or regenerate or self.nb_paths != nb_paths:
            self.paths = self.__generate_gbm(nb_paths=nb_paths)
        return self.paths

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


class GeometricBrownianMotionCandle(GeometricBrownianMotion):

    def __init__(self, volatility: float, sampling_rate: int = 10, drift: float = 0.0, initial_value: float = 1.0,
                 maturity: float = 1.0, time_intervals: int = 365):
        super().__init__(volatility=volatility, drift=drift, initial_value=initial_value,
                         maturity=maturity, time_intervals=time_intervals)

        # How many samples per (output) period
        self.sampling_rate = int(sampling_rate)

        # How many samples per (output) period
        self.time = TimeGenerator(maturity=maturity, intervals=time_intervals * self.sampling_rate)

    def generate(self, nb_paths: int = 1, regenerate: bool = True, rule="1D"):

        # Generate the paths according to our GBM methodology
        super().generate(nb_paths=nb_paths, regenerate=regenerate)
        df_high_sampling = self.to_datetime_index()

        # Reference the time origin
        time_origin = df_high_sampling.index.min()

        # Group the df for each path (OHLC)
        # into a dict indexed by path number (as int)
        paths = {i: df_high_sampling.iloc[:, i].resample(rule, origin=time_origin).ohlc() for i in range(nb_paths)}
        return paths


if __name__ == '__main__':
    # gbm = GeometricBrownianMotion(volatility=0.25)
    # gbm(10)

    gbm_ohlc = GeometricBrownianMotionCandle(volatility=0.25, sampling_rate=10)
    test = gbm_ohlc(10)
    print(type(test))
