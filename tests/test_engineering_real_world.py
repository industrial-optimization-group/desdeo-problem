from desdeo_problem.testproblems.EngineeringRealWorld import re21
from desdeo_problem.testproblems.EngineeringRealWorld import re22
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
def test_evaluate_problem():
    p: MOProblem = re21()

    xs = np.array([2, 2, 2, 2])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 1

    expected_result = np.array([[2048.528137, 0.02]])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.re22
def test_number_of_variabls():
    p: MOProblem = re22()

    assert p.n_of_variables == 3