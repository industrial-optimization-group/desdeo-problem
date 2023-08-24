from desdeo_problem.testproblems.EngineeringRealWorld import *
from desdeo_problem.problem import MOProblem
import pytest
import numpy as np
import numpy.testing as npt

@pytest.mark.re21
def test_number_of_variables_re21():
    p: MOProblem = re21()

    assert p.n_of_variables == 4

# Evaluating problem with some variable values
@pytest.mark.re21
def test_evaluate_re21():
    p: MOProblem = re21()

    # Variable values
    xs = np.array([2, 2, 2, 2])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 1

    expected_result = np.array([[2048.528137, 0.02]])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.re21
def test_variable_bounds_error_re21_1d():
    with pytest.raises(ValueError):
        p: MOProblem = re21(var_iv=np.array([1,2,3,4]))

@pytest.mark.re21
def test_variable_bounds_error_re21_2d():
    with pytest.raises(ValueError):
        p: MOProblem = re21(var_iv=np.array([[1,2,2,2],[4,1,2,0]]))

@pytest.mark.re21
def test_number_of_variables_error_re21_1d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re21(var_iv=np.array([]))

@pytest.mark.re21
def test_number_of_variables_error_re21_2d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re21(var_iv=np.array([[2,2,2],[2,2,2]]))

@pytest.mark.re22
def test_number_of_variables_re22():
    p: MOProblem = re22()

    assert p.n_of_variables == 3

# Evaluating problem with some variable values
@pytest.mark.re22
def test_evaluate_re22():
    p: MOProblem = re22()

    # Variable values
    xs = np.array([[6.37192567e+00, 1.44064899e+01, 4.57499269e-03]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 1

    expected_result = np.array([[185.84754575, 201.41659198]])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.re22
def test_variable_bounds_error_re22_1d():
    with pytest.raises(ValueError):
        p: MOProblem = re22(var_iv=np.array([1,-2,3]))

@pytest.mark.re22
def test_variable_bounds_error_re22_2d():
    with pytest.raises(ValueError):
        p: MOProblem = re22(var_iv=np.array([[1,2,2],[0,1,2]]))

@pytest.mark.re22
def test_number_of_variables_error_re22_1d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re22(var_iv=np.array([10,12,20,20]))

@pytest.mark.re22
def test_number_of_variables_error_re22_2d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re22(var_iv=np.array([[10,12,20,20],[10,12,20,20]]))

@pytest.mark.re23
def test_number_of_variables_re23():
    p: MOProblem = re23()

    assert p.n_of_variables == 4

# Evaluating problem with some variable values
@pytest.mark.re23
def test_evaluate_re23():
    p: MOProblem = re23()

    # Variable values
    xs = np.array([[42.28517847, 72.31212485, 10.02173122, 79.53649171]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 1

    expected_result = np.array([[   5211.1889184,  1266687.99823145]])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.re23
def test_variable_bounds_error_re23_1d():
    with pytest.raises(ValueError):
        p: MOProblem = re23(var_iv=np.array([1, 2, 3, 4]))

@pytest.mark.re23
def test_variable_bounds_error_re23_2d():
    with pytest.raises(ValueError):
        p: MOProblem = re23(var_iv=np.array([[50, 50, 201, 120], [0, -1, 12, 111]]))

@pytest.mark.re23
def test_number_of_variables_error_re23_1d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re23(var_iv=np.array([10,12,20]))

@pytest.mark.re23
def test_number_of_variables_error_re23_2d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re23(var_iv=np.array([[10,12,20],[10,12,20]]))


@pytest.mark.re24
def test_number_of_variables_re24():
    p: MOProblem = re24()

    assert p.n_of_variables == 2

# Evaluating problem with some variable values
@pytest.mark.re24
def test_evaluate_re24():
    p: MOProblem = re24()

    # Variable values
    xs = np.array([[ 1.95957702, 36.15606243]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 1

    expected_result = np.array([[4340.68706806,    0.        ]])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.re24
def test_variable_bounds_error_re24_1d():
    with pytest.raises(ValueError):
        p: MOProblem = re24(var_iv=np.array([1, 2]))

@pytest.mark.re24
def test_variable_bounds_error_re24_2d():
    with pytest.raises(ValueError):
        p: MOProblem = re24(var_iv=np.array([[2, 20], [0, -1]]))

@pytest.mark.re24
def test_number_of_variables_error_re24_1d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re24(var_iv=np.array([2]))

@pytest.mark.re24
def test_number_of_variables_error_re24_2d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re24(var_iv=np.array([[],[]]))

@pytest.mark.re25
def test_number_of_variables_re25():
    p: MOProblem = re25()

    assert p.n_of_variables == 3

# Evaluating problem with some variable values
@pytest.mark.re25
def test_evaluate_re25():
    p: MOProblem = re25()

    # Variable values
    xs = np.array([[29.77451832,  2.32877878,  0.09004689]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 1

    expected_result = np.array([[1.55630109e+00, 7.85132763e+06]])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.re25
def test_variable_bounds_error_re25_1d():
    with pytest.raises(ValueError):
        p: MOProblem = re25(var_iv=np.array([35, 0, 0.5]))

@pytest.mark.re25
def test_variable_bounds_error_re25_2d():
    with pytest.raises(ValueError):
        p: MOProblem = re25(var_iv=np.array([[10, 0.6, 0.2], [5, 13, 1]]))

@pytest.mark.re25
def test_number_of_variables_error_re25_1d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re25(var_iv=np.array([2, 3]))

@pytest.mark.re25
def test_number_of_variables_error_re25_2d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re25(var_iv=np.array([[],[]]))

@pytest.mark.re31
def test_number_of_variables_re31():
    p: MOProblem = re31()

    assert p.n_of_variables == 3

# Evaluating problem with some variable values
@pytest.mark.re31
def test_evaluate_re31():
    p: MOProblem = re31()

    # Variable values
    xs = np.array([[41.7022063,  72.03245214,  1.00022875]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 1

    expected_result = np.array([[273.82583798,   1.97697845, 273.72583798]])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.re31
def test_variable_bounds_error_re31_1d():
    with pytest.raises(ValueError):
        p: MOProblem = re31(var_iv=np.array([35, 0, 1.0]))

@pytest.mark.re31
def test_variable_bounds_error_re31_2d():
    with pytest.raises(ValueError):
        p: MOProblem = re31(var_iv=np.array([[10, 0.6, 2], [100, 13, 4]]))

@pytest.mark.re31
def test_number_of_variables_error_re31_1d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re31(var_iv=np.array([2, 3]))

@pytest.mark.re31
def test_number_of_variables_error_re31_2d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re31(var_iv=np.array([[],[]]))

@pytest.mark.re32
def test_number_of_variables_re32():
    p: MOProblem = re32()

    assert p.n_of_variables == 4

# Evaluating problem with some variable values
@pytest.mark.re32
def test_evaluate_re32():
    p: MOProblem = re32()

    # Variable values
    xs = np.array([[2.15798227, 7.23121249, 0.10113231, 1.59887129]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 1

    expected_result = np.array([[3.73662096e+01, 1.32736634e+03, 3.07903225e+07]])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.re32
def test_variable_bounds_error_re32_1d():
    with pytest.raises(ValueError):
        p: MOProblem = re32(var_iv=np.array([0.5, 0.5, 0, 5]))

@pytest.mark.re32
def test_variable_bounds_error_re32_2d():
    with pytest.raises(ValueError):
        p: MOProblem = re32(var_iv=np.array([[10, 0.6, 1, 0], [100, 13, 4, 1]]))

@pytest.mark.re32
def test_number_of_variables_error_re32_1d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re32(var_iv=np.array([2, 3]))

@pytest.mark.re32
def test_number_of_variables_error_re32_2d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re32(var_iv=np.array([[],[]]))

@pytest.mark.re33
def test_number_of_variables_re33():
    p: MOProblem = re33()

    assert p.n_of_variables == 4

# Evaluating problem with some variable values
@pytest.mark.re33
def test_evaluate_re33():
    p: MOProblem = re33()

    # Variable values
    xs = np.array([[  65.42555012,  100.21135727, 1000.22874963,   13.72099315]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 1

    expected_result = np.array([[3.59150353, 5.67635863, 0.        ]])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.re33
def test_variable_bounds_error_re33_1d():
    with pytest.raises(ValueError):
        p: MOProblem = re33(var_iv=np.array([0.5, 0.5, 0, 5]))

@pytest.mark.re33
def test_variable_bounds_error_re33_2d():
    with pytest.raises(ValueError):
        p: MOProblem = re33(var_iv=np.array([[10, 0.6, 1, 0], [100, 13, 4, 1]]))

@pytest.mark.re33
def test_number_of_variables_error_re33_1d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re33(var_iv=np.array([2, 3]))

@pytest.mark.re33
def test_number_of_variables_error_re33_2d():
    with pytest.raises(RuntimeError):
        p: MOProblem = re33(var_iv=np.array([[],[]]))
