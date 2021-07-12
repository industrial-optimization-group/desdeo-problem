"""A module for Lipschitzian regression

This module is based on BaseRegressor and can be used as an example for
building your own Regressor.
"""

import numpy as np
import pandas as pd

from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError


class LipschitzianRegressor(BaseRegressor):
    """Lipschitzian Regressor class

    Attributes:
        L (float): L -value: abs(f(a)-f(b)) =< L*abs(a-b)
        X (np.ndarray): points X in which we have data to calculate values f(x)
        y (np.ndarray): corresponding values of f(x)

    Arguments:
        L (float): L -value as above, default None
    """



    def __init__(self, L: float = None):
        self.L: float = L
        self.X: np.ndarray = None
        self.y: np.ndarray = None

    def fit(self, X, y):
        """Function to calculate L-value from data and save X and y.

        Arguments:
            X (np.ndarray or pd.DataFrame or pd.Series): Points X in which we
                    have data to calculate values f(x)
            y (np.ndarray or pd.DataFrame or pd.Series): corresponding values
                    of f(x)

        Raises:
            ModelError: if dimensions of X is not 1 or 2, or if dimensions of
                        y is not 1 or 2.

        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.reshape(-1, 1)

        # Make a 2-D array if needed
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self_dist_x = self.self_distance(X)
        self_dist_y = self.self_distance(y)
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.true_divide(self_dist_y, self_dist_x)
            delta[~np.isfinite(delta)] = -np.inf
            L = delta.max()
        self.X = X
        self.y = y
        self.L = L

    def predict(self, X):
        """Predict value of f(x) for given x with Lipscihitzian regression.

        For correct use X should correspond only a single point ie. be a
        single row. This is not tested at any point.

        Arguments:
            X (np.ndarray): X values for calculating prediction

        Returns:
            tuple(np.ndarray, np.ndarray): (y_mean, y_delta) which are mean
                    estimate of the f(y) for values in X and estimate of their
                    errors
        """
        dist = self.distance(X, self.X)
        y_low = (self.y - self.L * dist).max(axis=0)
        y_high = (self.y + self.L * dist).min(axis=0)
        y_mean = (y_low + y_high) / 2
        y_delta = np.abs((y_high - y_low) / 2)
        return (y_mean, y_delta)

    def self_distance(self, arr):
        """Calculates L1 distances between elements of given array.

        Arguments:
            arr (nd.array): 1d or 2d array

        Returns:
            nd.array: 1d or 2d array giving distances between elements of the
                    array calculated with L1 Norm

        Raises:
            ModelError: if arr is not 1d or 2d array.
        """
        if arr.ndim == 1:
            dist = np.abs(np.subtract(arr[None, :], arr[:, None]))
        elif arr.ndim == 2:
            dist = np.sum(np.abs(np.subtract(arr[None, :, :], arr[:, None, :])), axis=2)
        else:
            msg = (
                f"Array of wrong dimension. Expected dimension = 1 or 2. Recieved "
                f"dimension = {arr.ndim}"
            )
            raise ModelError(msg)
        return dist

    def distance(self, array1, array2):
        """Calculates L1 distances between elements of the input arrays

        Arrays must have matching dimenssions or they must be broadcastable to
        same shape

        Arguments:
            array1 (np.ndarray): array 1
            array2 (np.ndarray): array 2

        Returns:
            array: L1 distance between elements of input array after
                    broadcasting arrays to suitable shape.

        Raises:
            numpy related errors.
        """
        if array1.ndim == 1:
            array1 = array1.reshape(-1, 1)
        if array2.ndim == 1:
            array2 = array2.reshape(-1, 1)
        dist = np.sum(
            np.abs(np.subtract(array1[None, :, :], array2[:, None, :])), axis=2
        )
        return dist
