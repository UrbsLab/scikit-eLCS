import numpy as np
import time
import warnings
import sys
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Parallel, delayed

class eLCS(BaseEstimator):

    def __init__(self):
        """Sets up eLCS model with default parameters, and training data
        """


    def fit(self, X, y):
        """Scikit-learn required: Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels"""

    def transform(self,X):
        """Not needed for eLCS"""

    def fit_transform(self, X, y):
        self.fit(X,y)
        return self.transform(X)
