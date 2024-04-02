import numpy as np


class TimeGenerator(object):

    def __init__(self, maturity: float, intervals: int = 100):
        self.intervals = max(1, int(intervals))
        self.maturity = max(0.0, float(maturity))
        if self.maturity == 0.0:
            raise ValueError("Maturity must be positive.")
        
    @property
    def dt(self):
        return self.maturity / self.intervals

    @property
    def steps(self):
        return self.intervals + 1

    def __call__(self):
        return np.linspace(0.0, self.maturity, self.intervals + 1)


if __name__ == '__main__':
    t = TimeGenerator(maturity=1, intervals=12)  # 1 year of 12 months
    print(t())
