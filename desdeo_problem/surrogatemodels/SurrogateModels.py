"""Surrogate model module

These models are to be used as a surrogate for computationally heavy true
evaluation of the objective function.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR


class ModelError(Exception):
    """Raised when an error related to the surrogate models classes is encountered.

    """


# TODO: Add more sklearn methods here


class BaseRegressor(ABC):
    """Base Regressor class, abstract class with fit and predict methods.
    """



    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Abstract method for fitting the regressor

        Arguments:
            X (np.ndarray): values of explanatory variables
            y (np.ndarray): corresponding true values for result

        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Abstract method for predicting regression result

        Arguments:
            X (np.ndarray): Values of explanatory variables

        Return:
            Tuple[np.ndarray, np.ndarray]: Predicted values and
                                            their precisions
        """
        pass


class GaussianProcessRegressor(GPR, BaseRegressor):
    """GaussianProcessRegressor, from sklearn.

    Arguments:
        Look documentation from sklearn.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X: np.ndarray):
        """Predict values for GPR using sklearn function

        Arguments:
            X (np.ndarray): Values of explanatory variables.

        Return:
            Tuple[np.ndarray, np.ndarray]: Predicted values and
                                            their precisions

        """
        return super().predict(X, return_std=True)
