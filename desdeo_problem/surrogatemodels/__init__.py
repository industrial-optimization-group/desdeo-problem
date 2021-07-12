"""A sub-package of desdeo-problem for generating surrogate models

The surrogate models are used for replacing computationally heavy functions
with surrogate estimate which is easier to calculate.

"""

__all__ = ["ModelError", "BaseRegressor", "GaussianProcessRegressor", "LipschitzianRegressor"]

from desdeo_problem.surrogatemodels.lipschitzian import LipschitzianRegressor
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, GaussianProcessRegressor, ModelError
