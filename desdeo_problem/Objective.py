"""Defines Objective classes to be used in Problems

"""

import logging
import logging.config
from abc import ABC, abstractmethod
from os import path
from typing import Callable, Tuple, List, Union, NamedTuple

import numpy as np
import pandas as pd

from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError

log_conf_path = path.join(path.dirname(path.abspath(__file__)), "./logger.cfg")
logging.config.fileConfig(fname=log_conf_path, disable_existing_loggers=False)
logger = logging.getLogger(__file__)
# To prevent unexpected outputs in ipython console
logging.getLogger("parso.python.diff").disabled = True
logging.getLogger("parso.cache").disabled = True
logging.getLogger("parso.cache.pickle").disabled = True


class ObjectiveError(Exception):
    """Raised when an error related to the Objective class is encountered.

    """


# TODO consider replacing namedtuples with attr.s


class ObjectiveEvaluationResults(NamedTuple):
    """The return object of <problem>.evaluate methods.

    Attributes:
        objectives (Union[float, np.ndarray]): The objective function value/s for the
            input vector.
        uncertainity (Union[None, float, np.ndarray]): The uncertainity in the
            objective value/s.

    """

    objectives: Union[float, np.ndarray]
    uncertainity: Union[None, float, np.ndarray] = None

    def __str__(self):
        prnt_msg = (
            "Objective Evaluation Results Object \n"
            f"Objective values are: \n{self.objectives}\n"
            f"Uncertainity values are: \n{self.uncertainity}\n"
        )
        return prnt_msg


class ObjectiveBase(ABC):
    """The abstract base class for objectives.

    """

    def evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        """Evaluates the objective according to a decision variable vector.

        Args:
            variables (np.ndarray): A vector of Variables to be used in
            the evaluation of the objective.

        """
        return self.func_evaluate(decision_vector)

    @abstractmethod
    def func_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        """Evaluates the true objective value according to a decision variable vector.

        Uses the true (potentially expensive) evaluater if available. Otherwise,
        defaults to self.evaluate().

        Args:
            variables (np.ndarray): A vector of Variables to be used in
            the evaluation of the objective.

        """
        pass


class VectorObjectiveBase(ABC):
    """The abstract base class for multiple objectives which are calculated at once.

    """

    def evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        """Evaluates the objective according to a decision variable vector.

        Args:
            variables (np.ndarray): A vector of Variables to be used in
            the evaluation of the objective.

        """
        return self.func_evaluate(decision_vector)

    @abstractmethod
    def func_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        """Evaluates the true objective value according to a decision variable vector.

        Uses the true (potentially expensive) evaluater if available.

        Args:
            variables (np.ndarray): A vector of Variables to be used in
            the evaluation of the objective.

        """
        return self.evaluate(decision_vector)


class ScalarObjective(ObjectiveBase):
    """A simple objective function that returns a scalar.

    Args:
        name (str): Name of the objective.
        evaluator (Callable): The function to evaluate the objective's value.
        lower_bound (float): The lower bound of the objective.
        upper_bound (float): The upper bound of the objective.
        maximize (bool): Boolean to determine whether the objective is to be maximized.

    Attributes:
        name (str): Name of the objective.
        value (float): The current value of the objective function.
        evaluator (Callable): The function to evaluate the objective's value.
        lower_bound (float): The lower bound of the objective.
        upper_bound (float): The upper bound of the objective.
        maximize (List[bool]): List of boolean to determine whether the objectives are
            to be maximized. All false by default

    Raises:
        ObjectiveError: When ill formed bounds are given.

    """

    def __init__(
        self,
        name: str,
        evaluator: Callable,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
        maximize: List[bool] = [False],
    ) -> None:
        # Check that the bounds make sense
        if not (lower_bound < upper_bound):
            msg = ("Lower bound {} should be less than the upper bound " "{}.").format(
                lower_bound, upper_bound
            )
            logger.error(msg)
            raise ObjectiveError(msg)

        self.__name: str = name
        self.__evaluator: Callable = evaluator
        self.__value: float = 0.0
        self.__lower_bound: float = lower_bound
        self.__upper_bound: float = upper_bound
        self.maximize: bool = maximize  # TODO implement set/getters. Have validation.

    @property
    def name(self) -> str:
        return self.__name

    @property
    def value(self) -> float:
        return self.__value

    @value.setter
    def value(self, value: float):
        self.__value = value

    @property
    def evaluator(self) -> Callable:
        return self.__evaluator

    @property
    def lower_bound(self) -> float:
        return self.__lower_bound

    @property
    def upper_bound(self) -> float:
        return self.__upper_bound

    def func_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        """Evaluate the objective functions value.

        Args:
            variables (np.ndarray): A vector of variables to evaluate the
                objective function with.
        Returns:
            ObjectiveEvaluationResults: A named tuple containing the evaluated value,
                and uncertainity of evaluation of the objective function.

        Raises:
            ObjectiveError: When a bad argument is supplied to the evaluator.

        """
        try:
            result = self.__evaluator(decision_vector)
        except (TypeError, IndexError) as e:
            msg = "Bad argument {} supplied to the evaluator: {}".format(
                str(decision_vector), str(e)
            )
            logger.error(msg)
            raise ObjectiveError(msg)

        # Store the value of the objective
        self.value = result
        uncertainity = np.full_like(result, np.nan, dtype=float)
        # Have to set dtype because if the tuple is of ints, then this array also
        # becomes dtype int. There's no nan value of int type
        return ObjectiveEvaluationResults(result, uncertainity)


class VectorObjective(VectorObjectiveBase):
    """An objective object that calculated one or more objective functions.

    Args:
        name (List[str]): Names of the various objectives in a list
        evaluator (Callable): The function that evaluates the objective values
        lower_bounds (Union[List[float], np.ndarray), optional): Lower bounds of the
        objective values. Defaults to None.
        upper_bounds (Union[List[float], np.ndarray), optional): Upper bounds of the
        objective values. Defaults to None.
        maximize (List[bool]): *List* of boolean to determine whether the objectives are
            to be maximized. All false by default

    Raises:
        ObjectiveError: When lengths the input arrays are different.
        ObjectiveError: When any of the lower bounds is not smaller than the
        corresponding upper bound.

    """

    def __init__(
        self,
        name: List[str],
        evaluator: Callable,
        lower_bounds: Union[List[float], np.ndarray] = None,
        upper_bounds: Union[List[float], np.ndarray] = None,
        maximize: List[bool] = None,
    ):
        n_of_objectives = len(name)
        if lower_bounds is None:
            lower_bounds = np.full(n_of_objectives, -np.inf)
        if upper_bounds is None:
            upper_bounds = np.full(n_of_objectives, np.inf)
        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)
        # Check if list lengths are the same
        if not (n_of_objectives == len(lower_bounds)):
            msg = (
                "The length of the list of names and the number of elements in the "
                "lower_bounds array should be the same"
            )
            logger.error(msg)
            raise ObjectiveError(msg)
        if not (n_of_objectives == len(upper_bounds)):
            msg = (
                "The length of the list of names and the number of elements in the "
                "upper_bounds array should be the same"
            )
            logger.error(msg)
            raise ObjectiveError(msg)
        # Check if all lower bounds are smaller than the corresponding upper bounds
        if not (np.all(lower_bounds < upper_bounds)):
            msg = "Lower bounds should be less than the upper bound "
            logger.error(msg)
            raise ObjectiveError(msg)
        self.__name: List[str] = name
        self.__n_of_objectives: int = n_of_objectives
        self.__evaluator: Callable = evaluator
        self.__values: Tuple[float] = (0.0,) * n_of_objectives
        self.__lower_bounds: np.ndarray = lower_bounds
        self.__upper_bounds: np.ndarray = upper_bounds
        if maximize is None:
            self.maximize = [False] * n_of_objectives
        else:
            self.maximize: bool = maximize

    @property
    def name(self) -> str:
        return self.__name

    @property
    def n_of_objectives(self) -> int:
        return self.__n_of_objectives

    @property
    def values(self) -> Tuple[float]:
        return self.__values

    @values.setter
    def values(self, values: Tuple[float]):
        self.__values = values

    @property
    def evaluator(self) -> Callable:
        return self.__evaluator

    @property
    def lower_bounds(self) -> np.ndarray:
        return self.__lower_bounds

    @property
    def upper_bounds(self) -> np.ndarray:
        return self.__upper_bounds

    def func_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        """Evaluate the multiple objective functions value.

        Args:
            decision_vector (np.ndarray): A vector of variables to evaluate the
            objective function with.
        Returns:
            ObjectiveEvaluationResults: A named tuple containing the evaluated value,
                and uncertainity of evaluation of the objective function.

        Raises:
            ObjectiveError: When a bad argument is supplies to the evaluator or when
                the evaluator returns an unexpected number of outputs.

        """
        try:
            result = self.__evaluator(decision_vector)
        except (TypeError, IndexError) as e:
            msg = "Bad argument {} supplied to the evaluator: {}".format(
                str(decision_vector), str(e)
            )
            logger.error(msg)
            raise ObjectiveError(msg)
        result = tuple(result)

        # Store the value of the objective
        self.value = result
        uncertainity = np.full_like(result, np.nan, dtype=float)
        # Have to set dtype because if the tuple is of ints, then this array also
        # becomes dtype int. There's no nan value of int type
        return ObjectiveEvaluationResults(result, uncertainity)


class ScalarDataObjective(ScalarObjective):
    def __init__(
        self,
        name: List[str],
        data: pd.DataFrame,
        evaluator: Union[None, Callable] = None,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
        maximize: List[bool] = [False],
    ) -> None:
        if name in data.columns:
            super().__init__(name, evaluator, lower_bound, upper_bound, maximize)
        else:
            msg = f'Name "{name}" not found in the dataframe provided'
            raise ObjectiveError(msg)
        self.X = data.drop(name, axis=1)
        self.y = data[name]
        self.variable_names = self.X.columns
        self._model = None

    def train(
        self, model: BaseRegressor, index: List[int] = None, data: pd.DataFrame = None
    ):
        self._model = model
        if index is None and data is None:
            self._model.fit(self.X, self.y)
            return
        elif index is not None:
            self._model.fit(self.X[index], self.y[index])
            return
        elif data is not None:
            self._model.fit(data[self.variable_names], data[self.name])
            return
        msg = "I don't know how you got this error"
        raise ObjectiveError(msg)

    def evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        try:
            result, uncertainity = self._model.predict(decision_vector)
        except ModelError:
            msg = "Bad argument supplied to the model"
            raise ObjectiveError(msg)
        return ObjectiveEvaluationResults(result, uncertainity)

    def func_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        if self.__evaluator is None:
            msg = "No analytical function provided"
            raise ObjectiveError(msg)
        results = super().func_evaluate(decision_vector)
        self.X = np.vstack((self.X, decision_vector))
        self.y = np.vstack((self.y, results.objectives))
        return results


class VectorDataObjective(VectorObjective):
    def __init__(
        self,
        name: List[str],
        data: pd.DataFrame,
        evaluator: Union[None, Callable] = None,
        lower_bounds: Union[List[float], np.ndarray] = None,
        upper_bounds: Union[List[float], np.ndarray] = None,
        maximize: List[bool] = None,
    ) -> None:
        if all(obj in data.columns for obj in name):
            super().__init__(name, evaluator, lower_bounds, upper_bounds, maximize)
        else:
            msg = f'Name "{name}" not found in the dataframe provided'
            raise ObjectiveError(msg)
        self.X = data.drop(name, axis=1)
        self.y = data[name]
        self.variable_names = self.X.columns
        self._model = dict.fromkeys(name)  # TODO: Make the set of keys immutable?
        self._model_trained = dict.fromkeys(name, False)

    def train(
        self,
        models: Union[BaseRegressor, List[BaseRegressor]],
        index: List[int] = None,
        data: pd.DataFrame = None,
    ):
        if not isinstance(models, list):
            models = [models] * len(self.name)
        elif len(models) == 1:
            models = models * len(self.name)
        for model, name in zip(models, self.name):
            self.train_one_objective(name, model, index, data)

    def train_one_objective(
        self,
        name: str,
        model: BaseRegressor,
        index: List[int] = None,
        data: pd.DataFrame = None,
    ):
        if name not in self.name:
            raise ObjectiveError(
                f'"{name}" not found in the list of'
                f"original objective names: {self.name}"
            )
        self._model[name] = model
        if index is None and data is None:
            self._model[name].fit(self.X, self.y[name])
            self._model_trained[name] = True
            return
        elif index is not None:
            self._model[name].fit(self.X[index], self.y[name][index])
            self._model_trained[name] = True
            return
        elif data is not None:
            self._model[name].fit(data[self.variable_names], data[name])
            self._model_trained[name] = True
            return
        msg = "I don't know how you got this error"
        raise ObjectiveError(msg)

    def evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        if not all(self._model_trained.values()):
            msg = (
                f"Some or all models have not been trained.\n"
                f"Models for the following objectives have been trained:\n"
                f"{self._model_trained}"
            )
            raise ObjectiveError(msg)
        result = pd.DataFrame(index=range(decision_vector.shape[0]), columns=self.name)
        uncertainity = pd.DataFrame(
            index=range(decision_vector.shape[0]), columns=self.name
        )
        for name, model in self._model.items():
            try:
                result[name], uncertainity[name] = model.predict(decision_vector)
            except ModelError:
                msg = "Bad argument supplied to the model"
                raise ObjectiveError(msg)
        return ObjectiveEvaluationResults(result, uncertainity)

    def func_evaluate(self, decision_vector: np.ndarray) -> ObjectiveEvaluationResults:
        if self.__evaluator is None:
            msg = "No analytical function provided"
            raise ObjectiveError(msg)
        results = super().func_evaluate(decision_vector)
        self.X = np.vstack((self.X, decision_vector))
        self.y = np.vstack((self.y, results.objectives))
        return results
