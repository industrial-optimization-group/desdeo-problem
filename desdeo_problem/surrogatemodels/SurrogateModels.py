from abc import ABC, abstractmethod

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR


class ModelError(Exception):
    """Raised when an error related to the surrogate models classes is encountered.

    """


# TODO: Add more sklearn methods here


class BaseRegressor(ABC):
    @abstractmethod
    def fit(X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(X: np.ndarray) -> np.ndarray:
        pass


class GaussianProcessRegressor(GPR, BaseRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X: np.ndarray):
        return super().predict(X, return_std=True)
