from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes
from desdeo_problem.problem import MOProblem
import pytest
import numpy as np
import numpy.testing as npt

@pytest.mark.multiple_clutch_brakes
def test_number_of_variables():
    p: MOProblem = multiple_clutch_brakes()

    assert p.n_of_variables == 5

@pytest.mark.multiple_clutch_brakes
def test_number_of_objectives():
    p: MOProblem = multiple_clutch_brakes()

    assert p.n_of_objectives == 5

# Evaluating problem with some variable values
@pytest.mark.multiple_clutch_brakes
def test_evaluate_multiple_clutch_brakes():
    p: MOProblem = multiple_clutch_brakes()

    # Variable values
    xs = np.array([
        [80, 90, 2, 800, 8],
        [65, 85, 1.5, 980, 8],
        [73, 93, 1.5, 1000, 7]
    ])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 3

    expected_result = np.array([
        [0.74983533, 0.00528758059, 8, 90, 800],
        [0.99, 4.92, 8, 85, 980],
        [0.97, 4.98, 7, 93, 1000]
        ])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.multiple_clutch_brakes
def test_variable_bounds_error_1d():
    with pytest.raises(ValueError):
        p: MOProblem = multiple_clutch_brakes(var_iv=np.array([1,2,3,4,5]))

@pytest.mark.multiple_clutch_brakes
def test_variable_bounds_error_2d():
    with pytest.raises(ValueError):
        p: MOProblem = multiple_clutch_brakes(var_iv=np.array([[1,2,2,2,5],[1,2,3,4,1]]))

@pytest.mark.multiple_clutch_brakes
def test_number_of_variables_error_1d():
    with pytest.raises(RuntimeError):
        p: MOProblem = multiple_clutch_brakes(var_iv=np.array([]))

@pytest.mark.multiple_clutch_brakes
def test_number_of_variables_error_2d():
    with pytest.raises(RuntimeError):
        p: MOProblem = multiple_clutch_brakes(var_iv=np.array([[2,2,2],[2,2,2]]))