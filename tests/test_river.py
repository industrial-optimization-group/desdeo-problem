from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem.problem import MOProblem
import pytest
import numpy as np
import numpy.testing as npt

@pytest.mark.river_pollution
def test_problem():
    p: MOProblem = river_pollution_problem()

    assert p.n_of_variables == 2

@pytest.mark.river_pollution
def test_problem2():
    p: MOProblem = river_pollution_problem(five_obj=False)

    assert p.n_of_objectives == 4

@pytest.mark.river_pollution
def test_problem3():
    p: MOProblem = river_pollution_problem()

    assert p.n_of_objectives == 5

@pytest.mark.river_pollution
def test_problem4():
    p: MOProblem = river_pollution_problem()

    assert p.n_of_objectives == 4
