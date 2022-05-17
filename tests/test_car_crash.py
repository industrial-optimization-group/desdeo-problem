from desdeo_problem.testproblems.CarCrash import car_crash_problem
from desdeo_problem.problem import MOProblem
import pytest
import numpy as np
import numpy.testing as npt

@pytest.mark.car_crash
def test_number_of_variables():
    p: MOProblem = car_crash_problem()

    assert p.n_of_variables == 5

@pytest.mark.car_crash
def test_number_of_objectives():
    p: MOProblem = car_crash_problem()

    assert p.n_of_objectives == 3

@pytest.mark.car_crash
def test_car_crash():
    p: MOProblem = car_crash_problem()

    # evaluate the problem with some variable values
    xs = np.array([[2, 2, 2, 2, 2],[1, 2, 2, 2, 3]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([[1683.133345, 9.6266, 0.1233],[1685.231967, 9.4613, 0.1038]])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.car_crash
def test_variable_bounds_error():
    with pytest.raises(ValueError):
        p: MOProblem = car_crash_problem(var_iv=([2, 3, 0.9, 2, 4]))
