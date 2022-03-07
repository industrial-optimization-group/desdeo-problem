import numpy as np
import pytest
import matplotlib.pyplot as plt
from desdeo_problem.testproblems.DBMOPP.DBMOPP_generator import DBMOPP_generator


@pytest.fixture(scope="function")
def test_plots(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    problem.plot_problem_instance()
    po_set = problem.plot_pareto_set_members(100)
    assert len(po_set) > 0, "did not return po_set"
    problem.plot_landscape_for_single_objective(0, 100)
    # problem.plot_dominance_landscape(100) # not implemented yet


n_objectives = 5
n_variables = 4
n_local_pareto_regions = 2
n_dominance_res_regions = 1
n_global_pareto_regions = 3
const_space = 0.1
pareto_set_type = 2
constraint_type = 4
ndo = 1
neutral_space = 0.1

problem = DBMOPP_generator(
    n_objectives,
    n_variables,
    n_local_pareto_regions,
    n_dominance_res_regions,
    n_global_pareto_regions,
    const_space,
    pareto_set_type,
    constraint_type,
    ndo,
    False,
    False,
    neutral_space,
    10000,
)
print("Initializing works!")

x, point = problem.get_Pareto_set_member()

assert (
    x.any() or point.any() is not None
), "x or point are None in get_Pareto_set_member"

x = np.array(np.random.rand(5, n_variables))
moproblem = problem.generate_problem()

assert moproblem is not None, "moproblem was not formed"

# this does not work properly bc calls base class
# eval_res = moproblem.evaluate(x)
