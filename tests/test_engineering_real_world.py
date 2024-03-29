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
    xs = np.array([[10., 10., 20.], [12., 10., 20.], [11.5, 15, 35]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 3

    expected_result = np.array([[421.938, 2], [472.8, 2], [663.39, 164.0054642]])

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
    xs = np.array([[50, 50, 10, 10], [11, 63, 78, 187]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([[2996.845703, 5.9616],[49848.35467, 4266017.057]])

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
    xs = np.array([[2, 20], [3.3, 41.7]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([[2402, 3.63459881], [5007.3, 3.8568386109]])

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
    xs = np.array([[35.2, 15.5, 0.2], [1.5, 22.2, 0.4]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([[60.6336716, 34638.44539181], [34.0130175755, 494.270212155]])

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
    xs = np.array([[1.0, 2.0, 3.0], [62.3, 14.7, 1.2]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([[11.32455532, 33.3333333, 199924.503], 
                                [283.1345722, 1.117207759, 199991.79866]])

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
    xs = np.array([[1.0, 2.0, 3.0, 4.0],[2.5, 9, 3, 5]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([[11.44654, 0.0203259259, 18004862.58583042], 
                                [78.7378875, 0.01626074, 35163507.96891048]])

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
    xs = np.array([60, 110, 2557, 12])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 1

    expected_result = np.array([[4.5815, 2.4397401, 106196.71729885972]])

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
