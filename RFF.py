from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from scipy.stats import cauchy, laplace
import numpy as np
from scipy.spatial.distance import cdist
import scipy.io as sio
import math as math
import matplotlib.pyplot as plt


class RFF(BaseEstimator):
    def __init__(self, gamma = 1, D = 1, metric = "rbf"):
        self.gamma = gamma
        self.metric = metric
        # Dimensionality D (number of MonteCarlo samples)
        self.D = D
        self.fitted = False

    def fit(self, X, Xp, y=None):
        Z = self.RFF_main(X)
        Zp = self.RFF_main(Xp)
        return self.compute_kernel(Z.transform(X), Zp.transform(X))

    def RFF_main(self, X):
        """ Generates MonteCarlo random samples """
        d = X.shape[1]
        # Generate D iid samples from p(w)
        if self.metric == "rbf":
            self.w = np.sqrt(2 * self.gamma) * \
                np.random.normal(size=(self.D, d))
        elif self.metric == "laplace":
            self.w = cauchy.rvs(scale = self.gamma, size=(self.D, d))

        # Generate D iid samples from Uniform(0,2*pi)
        self.u = 2 * np.pi * np.random.rand(self.D)
        self.fitted = True
        return self

    def transform(self, X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        if not self.fitted:
            raise NotFittedError(
                "RBF_MonteCarlo must be fitted beform computing the feature map Z")
        # Compute feature map Z(x):
        Z = np.sqrt(2 / self.D) * \
            np.cos((X.dot(self.w.T) + self.u[np.newaxis, :]))
        return Z

    def compute_kernel(self, Z, Zp):
        """ Computes the approximated kernel matrix K """
        K = Z.dot(Zp.T)
        return K

    def get_params(self, deep=True):
        return {"gamma": self.gamma, "D": self.D, }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


