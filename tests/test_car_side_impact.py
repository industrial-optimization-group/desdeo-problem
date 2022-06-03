from xml.dom.expatbuilder import theDOMImplementation
from desdeo_problem.testproblems.CarSideImpact import car_side_impact
from desdeo_problem.problem import MOProblem
import pytest
import numpy as np
import numpy.testing as npt

@pytest.mark.car_side_impact
def test_number_of_variables():
    p: MOProblem = car_side_impact()

    assert p.n_of_variables == 7

@pytest.mark.car_side_impact
def test_number_of_objectives_3obj():
    p: MOProblem = car_side_impact()

    assert p.n_of_objectives == 3

@pytest.mark.car_side_impact
def test_number_of_objectives_4obj():
    p: MOProblem = car_side_impact(three_obj=False)

    assert p.n_of_objectives == 4

# Evaluating three-objective problem with some variable values
@pytest.mark.car_side_impact
def test_evaluate_car_side_impact_default():
    p: MOProblem = car_side_impact()

    # Variable values
    xs = np.array([[1, 1.2, 1.3, 1.4, 2.5, 0.6, 0.7], [1.5, 1.3, 1.4, 0.5, 0.9, 1.2, 1.2]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([[35.9330060000, 3.723599999, 11.852205], 
        [34.656012, 4.1242, 11.554582499]])

    npt.assert_allclose(objective_vectors, expected_result)

# Evaluating four-objective problem with some variable values
@pytest.mark.car_side_impact
def test_evaluate_car_side_impact_four_obj():
    p: MOProblem = car_side_impact(three_obj=False)

    # Variable values
    xs = np.array([[1, 1.2, 1.3, 1.4, 2.5, 0.6, 0.7], [1.5, 1.3, 1.4, 0.5, 0.9, 1.2, 1.2]])

    objective_vectors = p.evaluate(xs).objectives

    assert objective_vectors.shape[0] == 2

    expected_result = np.array([[35.9330060000, 3.723599999, 11.852205, 14.9884866000], 
        [34.656012, 4.1242, 11.554582499, 27.21423434000]])

    npt.assert_allclose(objective_vectors, expected_result)

@pytest.mark.car_side_impact
def test_variable_bounds_error_1d():
    with pytest.raises(ValueError):
        p: MOProblem = car_side_impact(var_iv=np.array([1,2,3,4,5,6,7]))

@pytest.mark.car_side_impact
def test_variable_bounds_error_2d():
    with pytest.raises(ValueError):
        p: MOProblem = car_side_impact(var_iv=np.array([[1,2,2,2,5,6,4],[1,2,3,4,1,2,0]]))

@pytest.mark.car_side_impact
def test_number_of_variables_error_1d():
    with pytest.raises(RuntimeError):
        p: MOProblem = car_side_impact(var_iv=np.array([]))

@pytest.mark.car_side_impact
def test_number_of_variables_error_2d():
    with pytest.raises(RuntimeError):
        p: MOProblem = car_side_impact(var_iv=np.array([[2,2,2],[2,2,2]]))