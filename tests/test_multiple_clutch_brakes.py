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
        [70, 90, 1.5, 1000, 3],
        [80, 110, 1.5, 1000, 9]
    ])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([
        [0.4704, 11.7617, 3, 90, 1000],
        [2.0948, 3.3505, 9, 110, 1000]
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