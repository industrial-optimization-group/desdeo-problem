import numpy as np
import pytest
from desdeo_problem.problem.Constraint import ConstraintError, ScalarConstraint

# helper functions

# Example of testing that ConstraintError works or custom errors in general
def decision_dims_wrong():
    s_const = ScalarConstraint("test_const", 3, 3, evaluator=evaluator)
    # 2 instead of needed 3
    decision_vector = np.array([1.0, 1.0])
    objective_vector = np.array([1.0, 1.0, 1.0])
    s_const.evaluate(decision_vector, objective_vector)


# just some testing how to test these. Mostly just running it through
def evaluator(x, y):
    res = x[0] + 7.7
    # how to do assertion checks
    assert len(y) > 0, "y empty"
    assert res == 8.7, "cal errr"
    return float(res)


# TESTS
def test_fails():
    with pytest.raises(ConstraintError):
        decision_dims_wrong()


def test_ScalarConstraint():
    s_const = ScalarConstraint("test_const", 3, 3, evaluator=evaluator)
    decision_vector = np.array([1.0, 1.0, 1.0])
    objective_vector = np.array([1.0, 1.0, 1.0])
    s_const.evaluate(decision_vector, objective_vector)
