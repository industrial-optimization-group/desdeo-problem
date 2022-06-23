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
    xs = np.array([[
    0.240, 7.128, 6.000, 5.573, 21.16, 85.20, 18.55, 3.738, 0.468,
    0.240, 7.150, 5.966, 5.564, 20.85, 85.21, 18.64, 3.750, 0.460,
    0.240, 7.186, 5.999, 5.563, 20.19, 85.45, 19.17, 3.750, 0.460
    ],
    [
    0.240, 7.000, 5.989, 5.575, 20.90, 85.13, 18.55, 3.749, 0.462,
    0.240, 7.028, 5.997, 5.575, 20.65, 85.22, 18.55, 3.749, 0.4600,
    0.240, 7.172, 6.000, 5.577, 20.40, 85.10, 18.66, 3.750, 0.460
    ],
    [
    0.240, 7.678, 4.611, 5.619, 22.76, 88.43, 18.30, 3.630, 0.514,
    0.240, 7.770, 4.848, 5.597, 23.84, 88.34, 16.02, 3.620, 0.587,
    0.241, 7.650, 4.420, 5.592, 23.06, 89.19, 15.44, 3.623, 0.508
    ]
    ])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 3

    expected_result = np.array([[
        73.5, 1980, 62, 2.000, 435, 44000, -2450, -15.3, -188, 0.3
        ],
        [
        73.5, 1960, 62, 2.000, 435, 44000, -2450, -15.3, -190, 0.1
        ],
        [
        73.7, 1880, 60, 2.000, 465, 42000, -2050, -15.8, -200, 0.8
        ]
    ])

    npt.assert_allclose(objective_vectors, expected_result)

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