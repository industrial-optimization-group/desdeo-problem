from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem.problem import MOProblem
import pytest

@pytest.mark.river_pollution
def test_problem():
    p: MOProblem = river_pollution_problem()

    assert p.n_of_variables == 2