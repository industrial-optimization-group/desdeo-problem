from desdeo_problem.testproblems.EngineeringRealWorld import *
# kysy importataanko kaikki re... funktiot samaan testitiedostoon
from desdeo_problem.problem import MOProblem
import pytest
import numpy as np
import numpy.testing as npt

@pytest.mark.re21
def test_number_of_variables():
    p: MOProblem = re21()

    assert p.n_of_variables == 4

@pytest.mark.re21
def test_evaluate_re21():
    p: MOProblem = re21()

    xs = np.array([2, 2, 2, 2])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 1

    expected_result = np.array([[2048.528137, 0.02]])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.re21
def test_variable_bounds_error_1d():
    with pytest.raises(ValueError):
        p: MOProblem = re21(var_iv=np.array([1,2,3,4]))

@pytest.mark.re21
def test_variable_bounds_error_2d():
    with pytest.raises(ValueError):
        p: MOProblem = re21(var_iv=np.array([[1,2,2,2],[4,1,2,0]]))

@pytest.mark.re22
def test_number_of_variabls():
    p: MOProblem = re22()

    assert p.n_of_variables == 3

@pytest.mark.re22
def test_evaluate_re22():
    p: MOProblem = re22()

    xs = np.array([[10, 10, 20], [12, 10, 20]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 1

    expected_result = np.array([[414, 2], [472.8, 2]])

    npt.assert_allclose(objective_vectors, expected_result)