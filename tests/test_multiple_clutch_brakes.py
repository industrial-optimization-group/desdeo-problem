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
    # Values form Osyczka, A. (1992)
    xs = np.array([
        [.5854315e2, .7867608e2, .1499999e1, .9002973e3, .3000000e1],
        [.5583733e2, .1057214e3, .2480044e1, .7716822e3, .3000000e1]
    ])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([
        [.4061785e0, .1495035e2, .3000000e1, .7867608e2, .9002973e3],
        [.1959098e1, .1447586e2, .3000000e1, .1057214e3, .7716822e3]
        ])

    # rtol because difference in floating point number calculation in old fortran 
    npt.assert_allclose(objective_vectors, expected_result, rtol=1e-4)

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