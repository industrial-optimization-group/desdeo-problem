from desdeo_problem.testproblems.GAA import gaa
from desdeo_problem.problem import MOProblem
import pytest
import numpy as np
import numpy.testing as npt

@pytest.mark.gaa
def test_number_of_variables():
    p: MOProblem = gaa()

    assert p.n_of_variables == 27

@pytest.mark.gaa
def test_number_of_objectives():
    p: MOProblem = gaa()

    assert p.n_of_objectives == 10

# Evaluating problem with some variable values
@pytest.mark.gaa
def test_evaluate_gaa():
    p: MOProblem = gaa()

    # Variable values
    # Values from Woodruff, Reed & Simpson (2013) table 6 solution C and D
    xs = np.array([
        [0.24, 7.768, 4.611, 5.619, 22.76, 88.43, 18.30, 3.63, 0.514,
        0.24, 7.77, 4.848, 5.597, 23.84, 88.34, 16.02, 3.62, 0.587,
        0.241, 7.65, 4.42, 5.592, 23.06, 89.19, 15.44, 3.623, 0.508],
        [0.268, 7.094, 1.468, 5.509, 22.62, 89.25, 19.23, 3.657, 0.566,
        0.268, 7.181, 1.289, 5.502, 22.48, 88.47, 18.97, 3.635, 0.547,
        0.275, 7.075, 1.482, 5.510, 22.48, 89.03, 19.39, 3.644, 0.563]
    ])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([
        [73.7, 1880, 60, 2, 465, 42000, -2050, -15.8, -200, 0.8],
        [73.3, 1960, 72, 1.925, 435, 44000, -2150, -14.9, -192, 0.2]
    ])

    # rtol because old PFPF in f_10 was replaced
    npt.assert_allclose(objective_vectors, expected_result, rtol=2)

@pytest.mark.gaa
def test_variable_bounds_error_1d():
    with pytest.raises(ValueError):
        p: MOProblem = gaa(var_iv=np.array([
    0.36, 9, 3, 5.7, 22, 97, 17, 3.4, 0.75,
    0.36, 9, 3, 5.7, 22, 97, -17, 3.4, 0.75,
    0.36, 9, 3, 5.7, 22, 97, 17, 3.4, 0.75
    ]))

@pytest.mark.gaa
def test_variable_bounds_error_2d():
    with pytest.raises(ValueError):
        p: MOProblem = gaa(var_iv=np.array([[
    0.36, 9, 3, 5.7, 22, 97, 17, 3.4, 0.75,
    0.36, 9, 3, 5.7, 22, 97, 17, 3.4, 0.75,
    0.36, 9, 3, 5.7, 22, 97, 17, 3.4, 0.75
    ], 
    [0.36, 9, 3, 5.7, 22, 97, 17, 3.4, 0.75,
    0.36, 9, 3, 5.7, 22, 97, 17, 3.4, 0.75,
    0, 9, 3, 5.7, 22, 97, 17, 3.4, 0.75
    ]]))

@pytest.mark.gaa
def test_number_of_variables_error_1d():
    with pytest.raises(RuntimeError):
        p: MOProblem = gaa(var_iv=np.array([]))

@pytest.mark.gaa
def test_number_of_variables_error_2d():
    with pytest.raises(RuntimeError):
        p: MOProblem = gaa(var_iv=np.array([[2,2,2],[2,2,2]]))