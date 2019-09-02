"""Defines Objective classes to be used in Problems

"""

import logging
import logging.config
from abc import ABC, abstractmethod
from os import path
from typing import Callable, Tuple, List, Union

import numpy as np

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


class ObjectiveBase(ABC):
    """The abstract base class for objectives.

    """

    @abstractmethod
    def evaluate(self, decision_vector: np.ndarray) -> float:
        """Evaluates the objective according to a decision variable vector.

        Args:
            variables (np.ndarray): A vector of Variables to be used in
            the evaluation of the objective.

        """
        pass


class VectorObjectiveBase(ABC):
    """The abstract base class for multiple objectives which are calculated at once.

    """

    @abstractmethod
    def evaluate(self, decision_vector: np.ndarray) -> Tuple[float]:
        """Evaluates the objective according to a decision variable vector.

        Args:
            variables (np.ndarray): A vector of Variables to be used in
            the evaluation of the objective.

        """
        pass


class ScalarObjective(ObjectiveBase):
    """A simple objective function that returns a scalar.

    Args:
        name (str): Name of the objective.
        evaluator (Callable): The function to evaluate the objective's value.
        lower_bound (float): The lower bound of the objective.
        upper_bound (float): The upper bound of the objective.

    Attributes:
        name (str): Name of the objective.
        value (float): The current value of the objective function.
        evaluator (Callable): The function to evaluate the objective's value.
        lower_bound (float): The lower bound of the objective.
        upper_bound (float): The upper bound of the objective.

    Raises:
        ObjectiveError: When ill formed bounds are given.

    """

    def __init__(
        self,
        name: str,
        evaluator: Callable,
        lower_bound: float = -np.inf,
        upper_bound: float = np.inf,
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

    def evaluate(self, decision_vector: np.ndarray) -> float:
        """Evaluate the objective functions value.

        Args:
            variables (np.ndarray): A vector of variables to evaluate the
            objective function with.
        Returns:
            float: The evaluated value of the objective function.

        Raises:
            ObjectiveError: When a bad argument is supplies to the evaluator.

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

        return result


class VectorObjective(VectorObjectiveBase):
    """An objective object that calculated one or more objective functions.

    Args:
        name (List[str]): Names of the various objectives in a list
        evaluator (Callable): The function that evaluates the objective values
        lower_bounds (Union[List[float], np.ndarray), optional): Lower bounds of the
        objective values. Defaults to None.
        upper_bounds (Union[List[float], np.ndarray), optional): Upper bounds of the
        objective values. Defaults to None.

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

    def evaluate(self, decision_vector: np.ndarray) -> Tuple[float]:
        """Evaluate the multiple objective functions value.

        Args:
            decision_vector (np.ndarray): A vector of variables to evaluate the
            objective function with.
        Returns:
            float: The evaluated value of the objective function.

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
        if not (len(result) == self.n_of_objectives):
            msg = (
                "Number of output ({}) elements not equal to the expected "
                "number of output elements ({})".format(
                    str(result), self.n_of_objectives
                )
            )
            logger.error(msg)
            raise ObjectiveError(msg)
        # Store the value of the objective
        self.value = result
        return result
