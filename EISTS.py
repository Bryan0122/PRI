from sklearn.base import BaseEstimator
import numpy as np
import scipy.io as sio
import math as math


class EISTS(BaseEstimator):

    def __init__(self, r=40, sigma=0.01):
        self.r = r
        self.sigma = sigma

    def fit(self, X, Xp):
        self.X = X
        self.Xp = Xp
        self.K = self.compute_kernel()
        return self

    def compute_kernel(self):
        K = np.exp(-(self.X.dot(self.X.T) + self.Xp.dot(self.Xp.T)
                     ) / (2 * self.sigma**2)) * self.CTS()
        return K

    def CTS(self):
        num = self.X.dot(self.Xp.T)
        re = np.zeros(num.shape)
        for i in range(0, self.r):
            re = re + (1 / math.factorial(i) * (num / self.sigma**2)**i)
        return re

    def get_params(self, deep=True):
        return {"r": self.r, "sigma": self.sigma, }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self



