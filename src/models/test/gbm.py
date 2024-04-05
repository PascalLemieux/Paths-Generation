import unittest

import numpy as np

from src.models.gbm import GeometricBrownianMotion


class GeometricBrownianMotionTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_avg_no_drift(self):

        means, spot = [], 100.0

        for i in range(100):
            np.random.seed(i)
            gbm = GeometricBrownianMotion(volatility=0.5, drift=0.0, initial_value=spot)(10000)
            exp_maturity = gbm.iloc[-1, :].mean()
            means.append(exp_maturity)

        print(f"TEST: 'test_avg_no_drift' result is {np.mean(means):.4f} vs {spot:.4f}.")
        test_result = np.abs(np.mean(means) / spot - 1.0)
        self.assertAlmostEqual(test_result, 0.0, 1)

    def test_avg_pos_drift(self):

        drift, maturity = 0.1, 1.0
        means, spot = [], 100.0
        forward = spot * np.exp(drift * maturity)

        for i in range(100):
            np.random.seed(i)
            gbm = GeometricBrownianMotion(volatility=0.5, drift=drift, maturity=maturity, initial_value=100.0)(10000)
            exp_maturity = gbm.iloc[-1, :].mean()
            means.append(exp_maturity)

        print(f"TEST: 'test_avg_pos_drift' result is {np.mean(means):.4f} vs {forward:.4f}.")
        test_result = np.abs(np.mean(means) / forward - 1.0)
        self.assertAlmostEqual(test_result, 0.0, 1)

    def test_avg_pos_drift_long_maturity(self):

        drift, maturity = 0.1, 10.0  # 10 years
        ts = int(maturity * 52)  # once a week

        means, spot = [], 100.0
        forward = spot * np.exp(drift * maturity)

        for i in range(100):
            np.random.seed(i)
            gbm = GeometricBrownianMotion(volatility=0.5, drift=drift, maturity=maturity,
                                          time_intervals=ts, initial_value=100.0)(10000)
            exp_maturity = gbm.iloc[-1, :].mean()
            means.append(exp_maturity)

        print(f"TEST: 'test_avg_pos_drift_long_maturity' result is {np.mean(means):.4f} vs {forward:.4f}.")
        test_result = np.abs(np.mean(means) / forward - 1.0)
        self.assertAlmostEqual(test_result, 0.0, 1)
