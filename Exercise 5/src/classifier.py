import numpy as np
import pandas as pd
from scipy.stats import mode


class AwesomeClassifier():
    def __init__(self):
        self._prediction = 0

    def fit(self, X, y):
        self._prediction = mode(y)[0]
        return self

    def predict(self, X):
        return np.full(len(X), self._prediction)
