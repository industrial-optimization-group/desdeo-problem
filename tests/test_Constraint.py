import numpy as np
import pytest
from desdeo_problem.problem.Constraint import ScalarConstraint, ConstraintBase


# @pytest.fixture
# just some testing how to test these. Mostly just running it through
def evaluator(x, y):
    res = x[0] + 7.7
    assert res == 8.7, "cal errr"
    return float(res)


s_const = ScalarConstraint("test_const", 3, 3, evaluator=evaluator)

decision_vector = np.array([1.0, 1.0, 1.0])
objective_vector = np.array([1.0, 1.0, 1.0])
res = s_const.evaluate(decision_vector, objective_vector)
